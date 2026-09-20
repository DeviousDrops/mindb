# Architectural decisions

A log of every non-obvious design choice in MinDB, in the order a reader
would hit them working top-down through the system: concurrency, memory
layout, the search algorithm, quantization, durability, and the wire
protocol. Each entry is **decision → why → what it cost**. Where a decision
reversed an earlier one, the earlier approach and why it was rejected is
included, because the rejection is often more instructive than the choice.

This complements [`ARCHITECTURE.md`](ARCHITECTURE.md), which is the
narrative design doc with measurements; this file is the flat list, meant to
be skimmed or grepped. For what the code actually does rather than why, see
[`FEATURES.md`](FEATURES.md); for the pitch and the numbers, see
[`README.md`](../README.md).

---

## Concurrency

### Decision: one `sync.RWMutex`, held across the whole scan

**Why:** `RLock`/`RUnlock` costs ~20 ns; a search costs ~1–6 ms. The lock is a
rounding error next to the work it protects, so there is nothing to win by
avoiding it.

**Rejected alternative: lock-free reads via tombstoning + RCU compaction.**
This was the original design and it was implemented far enough to reveal
three real bugs before being thrown out:

1. **Hard crash.** `idMap` is a plain Go map. Two concurrent `Insert`s, or an
   `Insert` racing a `Delete`, hit the Go runtime's `concurrent map read and
   map write` — an unrecoverable *throw*, not a `panic`. The process dies.
2. **Silent corruption.** `Insert` wrote `dims` float32s into the vector array
   and then flipped a `live` bit, with no memory barrier between the two. A
   concurrent searcher could read a half-written vector and return a
   plausible *wrong* score — no error, no crash, just a bad answer.
3. **Lost writes.** RCU compaction built a fresh copy of the engine in the
   background and swapped a pointer. Writes landing on the *old* copy during
   the swap vanished. Real RCU is "lock-free readers, synchronized writers" —
   it never promised lock-free writers, and the design had quietly assumed it
   did.

There was also a fourth problem that wasn't a bug so much as a wasted
subsystem: capacity is preallocated, so compaction never actually reclaimed
memory — it only made the live set denser. An entire background goroutine,
a second full copy of the engine, and a transient 2x memory spike existed to
solve a problem a ten-line free list solves with zero concurrency exposure.

**Cost of the RWMutex approach:** a writer waits behind an in-flight scan.
Concurrent searches still run fully in parallel — `RLock` is shared, only
`Lock` (writes) is exclusive. For a read-dominated sidecar this is the right
trade, and unlike the lock-free version it is *actually correct*.

---

### Decision: replace RCU compaction with a free list

**Why:** capacity is fixed and preallocated at `New(dims, capacity)`, so
"compaction" was never reclaiming memory — the backing arrays are the same
size before and after. A `Delete` pushes the freed slot index onto
`e.free []uint32`; the next `Insert` pops from it before growing
`highWater`. No background goroutine, no second copy, no pointer swap.

**Consequence:** a vector's internal slot index is *not* stable — deleting
and re-inserting can reuse a slot. External IDs are stable; slots are an
implementation detail (`pkg/core/engine.go`, `idMap`).

---

## Memory layout

### Decision: struct-of-arrays (`Engine` holds parallel slices, not `[]Vector`)

**Why:** a scan only needs `vectors` (or `codes`+`scales`+`residuals` for the
cascade) — it never touches `externalID` or `payloads` until after top-k is
decided. Struct-of-arrays means every cache line the scan pulls in from
`vectors` is 100% useful data, versus array-of-structs where each cache line
would carry ID and payload bytes the hot loop doesn't need.

**Cost:** more bookkeeping arrays to keep in sync (`live`, `idMap`, `free`,
`highWater` all have to agree), and `Delete` has to touch several slices
instead of one. Justified because the scan is the operation being optimized
for; deletes are comparatively rare and cheap regardless.

### Decision: normalize every vector to unit length at insert

**Why:** if `‖v‖ = ‖q‖ = 1`, cosine similarity **is** the dot product — no
division, no magnitude array, no extra random-access load in the hot loop.
It is also a *precondition* of the cascade's error bound (see below):
Cauchy-Schwarz gives `|q·(v−v̂)| ≤ ‖q‖·ρ`, and that collapses to `≤ ρ` only
because `‖q‖ = 1`.

**Cost:** one `Normalize` pass at insert (`O(dims)`, done outside the lock)
and one at query time. `Insert` copies the caller's vector before
normalizing so the caller's slice is never mutated
(`pkg/core/engine.go:Insert`).

### Decision: bounded min-heap for top-k, not a sort

**Why:** sorting all live scores is `O(n log n)` plus an allocation for `n`
in the tens of thousands — enough to erase the gains from everything else.
A size-`k` min-heap (`pkg/core/engine.go:topK`) does `O(log k)` work per
candidate and the common case (`score` below the current worst survivor) is
one predictable branch that rejects the candidate outright. Heaps are pooled
per-worker via `sync.Pool` and merged after a parallel scan.

### Decision: parallel scan across `GOMAXPROCS`, gated by a size threshold

**Why:** fan-out and per-worker heap merge cost more than they save below
`parallelScanThreshold = 8192` live vectors (`pkg/core/engine.go`), so small
engines scan on a single goroutine. Above that, the corpus is split into
contiguous chunks, each worker gets a pooled heap, and the heaps are merged
at the end.

### Decision: ties break on slot index during scan, on external ID during presentation

**Why:** parallelizing the scan means candidates with equal scores can arrive
in different orders depending on how many workers ran, so the heap's
internal tiebreak (lower slot wins, `worse()` in `engine.go`) exists purely
to make output deterministic regardless of `GOMAXPROCS` — the differential
test compares scores exactly and would otherwise be flaky. The *final*
sort in `Search` breaks ties on external ID instead, because slot indices
are not stable across a snapshot reload and IDs are.

---

## Search: the bound-and-refine cascade

### Decision: store both a float32 vector and an int8 quantized copy, plus a residual norm

**Why:** this is the core idea of the project. Alongside the exact float32
vector, every insert also computes an int8 code, a per-vector scale, and a
residual norm `ρ = ‖v − v̂‖` (`pkg/math/quantize.go:Quantize`). A search first
scores every vector cheaply against its int8 code (a quarter of the bytes),
then uses `ρ` to get a *hard* two-sided bound on the true score via
Cauchy-Schwarz — not a heuristic estimate. See `docs/ARCHITECTURE.md` for
the full derivation.

**Why this differs from Qdrant/Weaviate/Milvus-style quantization:** those
systems score with compressed codes, oversample by a constant factor,
rescore, and hope the true top-k survived — recall is a property you
*measure*, not one you're guaranteed. MinDB's pruning is provably lossless:
a vector is discarded only when it is mathematically impossible for it to be
in the top-k. The result is bit-identical to brute force, and a differential
test in `pkg/core/cascade_test.go` (`TestCascadeIsExact`) asserts exactly
that across thousands of random queries.

### Decision: asymmetric scoring — the query is never quantized

**Why:** an earlier version considered quantizing both sides (symmetric
scoring), which was found unsound: the error bound would need to account for
error in *both* the query and the database vector, which is a much looser
bound and, worse, was derived incorrectly in the first pass (documented as a
retired claim in `docs/ARCHITECTURE.md`). Keeping the query in float32 means
the only error term is the stored vector's own residual, giving a
measurably tighter bound (4.3x, per `ARCHITECTURE.md`) — and it means the
SIMD kernel only needs `VPMOVSXBD`/`VCVTDQ2PS`/`VFMADD231PS` (AVX2 + FMA3),
not the AVX-512 VNNI instructions that true int8×int8 dot products need.

### Decision: residual norm is measured directly, not derived from the scale

**Why:** an earlier version computed `‖v̂‖ = √(1 − ρ²)`, which assumes the
quantization residual is orthogonal to the reconstruction. int8 rounding is
not an orthogonal projection, so that identity is only approximate — and an
approximate bound isn't a proof. `Quantize` now does one extra pass at
insert time to measure `ρ` directly (`pkg/math/quantize.go`). Cost: one
extra `O(dims)` pass per insert, never at query time.

### Decision: guard fallback to a plain float32 scan when survivors exceed 25% of the corpus

**Why:** the cascade's win depends on most vectors being prunable in pass 1.
If pass 2's survivor set gets large, pass 3 (scattered exact rescoring, which
walks the float32 array out of order and runs serially) can end up costing
more than a straight parallel float32 scan would have. `guardFraction = 0.25`
(`pkg/core/cascade.go`) was set from `BenchmarkGuardCrossover`, which found
the crossover (scattered rescore alone equalling a full scan) at ~49%
survivors on this machine; 0.25 sits at the cautious end of the plausible
range once the bound pass itself is accounted for. Both paths return
identical results, so getting this constant wrong only costs time, never
correctness — and in practice it's nearly unreachable, since clustered data
measures ~0.4% survivors, two orders of magnitude under the threshold.

### Decision: the cascade only activates when `math.HasFastInt8()` is true

**Why:** pure-Go int8 dot products plateau at ~0.83 G MAC/s — Go cannot emit
the packed multiply-accumulate instruction that makes int8 arithmetic pay
off, across every unroll factor tried (`docs/ARCHITECTURE.md`). Without SIMD,
scoring int8 codes is *slower* than just scanning float32 directly, so
taking the cascade path would make searches slower, not faster.
`Engine.useCascade` defaults to `math.HasFastInt8()` and can be overridden
per-engine via `SetCascade` for benchmarking.

---

## The AVX2 kernel (stage 3)

### Decision: generate the assembly with Avo rather than hand-write it

**Why:** hand-written Plan9 assembly is easy to get subtly wrong (register
allocation, calling convention, stack frame size) and hard to review. Avo
(`github.com/mmcloughlin/avo`) is a Go DSL that emits Plan9 assembly plus a
matching Go stub, so the source of truth
(`pkg/math/avo/asm.go`) is normal, readable Go, and the generated
`pkg/math/dotint8_avx2_amd64.s` / `dotint8_avx2_stub_amd64.go` are committed artifacts
regenerated via `go run asm.go -out ... -stubs ...`.

**Why AVX2 + FMA3, not AVX-512:** AVX2 is available on every x86-64 CPU since
Haswell (2013) / Zen (2017); AVX-512 is absent on most consumer Zen ≤3 parts
and disabled on Intel's hybrid consumer lineups, and suffers downclocking on
several server parts. Because scoring is asymmetric (query stays float32),
the kernel never needs int8×int8 multiply, which is the operation AVX-512
VNNI exists for — `VPMOVSXBD` (sign-extend 8×int8 → 8×int32) +
`VCVTDQ2PS` (convert → float32) + `VFMADD231PS` (multiply-accumulate against
the float32 query) covers it entirely on AVX2.

**Why `golang.org/x/sys/cpu` for detection, not a `cpuid` call written by
hand:** it's already an indirect dependency (pulled in by gRPC's transitive
graph) and is the standard, well-tested way to read CPUID flags in Go.
`pkg/math/kernel_amd64.go`'s `init()` checks `HasAVX2 && HasFMA && HasAVX`
and only then swaps `dotInt8Impl` to the generated kernel; every other
platform (including non-amd64 entirely, via `kernel_generic.go`) falls back
to `dotInt8Generic`, so MinDB is correct everywhere and fast where the
hardware allows it.

**Why Avo is declared as a `tool` dependency in `go.mod`, not a normal
`require`:** the generator (`pkg/math/avo/asm.go`) carries a
`//go:build ignore` tag so it's excluded from every normal build — otherwise
`go build ./...` would see two `package main`/`package math` files fighting
over the same directory. Because nothing in the buildable graph imports Avo,
`go mod tidy` would otherwise drop it from `go.mod` entirely, breaking
regeneration for the next person. Go 1.24's `tool` directive
(`go get -tool github.com/mmcloughlin/avo/build`) pins it explicitly for
exactly this case: a dependency only `go run`/`go generate` needs, never
`go build`.

**Measured impact** (this machine, `BenchmarkSearchPaths`, N=20,000,
dims=768): brute force 6.47 ms, cascade with the AVX2 kernel active 1.75 ms
— a real, measured 3.7x, not a projection. The raw kernel itself sustains
~23.5 GB/s at 768 dims, comfortably above this machine's ~11.7 GB/s memory
bandwidth, meaning the int8 scan is now bandwidth-bound rather than
compute-bound — exactly the point of writing the assembly in the first
place.

---

## Portability

### Decision: the generated AVX2 stub is constrained by filename *and* an explicit build tag

**Why:** the stub was called `dotint8_avx2_amd64_stub.go`, which looks
architecture-constrained and is not. Go derives the implicit `GOARCH`
constraint from the filename component immediately before `.go`, so `_amd64`
in the middle of the name does nothing: the bodyless `func dotInt8AVX2`
declaration was compiled on every architecture, with no `.s` file to satisfy
it, and `GOARCH=arm64 go build ./...` failed with `missing function body`.
MinDB did not build for its own deploy target.

Renamed to `dotint8_avx2_stub_amd64.go`, which does carry the constraint, plus
an explicit `//go:build amd64`. The filename alone is sufficient; the tag is
there because the filename convention is exactly what failed to be noticed the
first time, and a tag is legible to a reader who does not have the rule
memorized.

**Cost:** avo does not emit build tags, so `make asm` re-adds it after
regenerating. A bare `go generate` drops the tag and keeps the build correct
anyway, since the filename is doing the real work.

**What actually prevents a recurrence:** `make check-cross` vets amd64, arm64
and riscv64. This class of bug is invisible to a native build, which is how it
survived in the first place — no test could have caught it, only a
cross-build.

### Decision: three kernel files — amd64, arm64, generic — not amd64 and a `!amd64` catch-all

**Why:** arm64 reached the pure-Go fallback by omission, which reads as an
oversight rather than a decision, and left nowhere obvious for a NEON kernel to
go. `kernel_arm64.go` is byte-identical in behaviour to `kernel_generic.go`
today and earns its place by being the file a NEON implementation drops into,
already wired to `dotInt8Impl` and already registered with the cross-check.

**Cost:** one duplicated three-line const block. Cheap enough that the
alternative — a comment in the catch-all saying "arm64 also lands here" --
buys nothing.

### Decision: the cross-check compares every reachable kernel against `dotInt8Generic`, with a tolerance scaled by the sum of absolute terms

**Why:** `dotInt8Generic` is the oracle: it defines a correct result and every
other path is an optimization of it. Per-architecture
`kernel_variants_*_test.go` files register whatever SIMD kernels the build
compiled in, so the test on arm64 is tautological today and becomes a real
differential test the moment a NEON kernel is registered — without anyone
having to remember to write it then.

**Why not a tolerance relative to the result:** a dot product over mixed-sign
terms cancels, so the result can be near zero while the terms summed to reach
it are large. A result-relative bound would then demand precision no float32
accumulation can deliver. The error scales with `sum |q_i * code_i|`, so the
bound does too: `1e-5 * sumAbs + 1e-6`, roughly twice the worst case for the
8-accumulator pure-Go loop at a 2^-24 unit roundoff. Measured headroom is
~2000x over the real AVX2-vs-generic margin at dims=768, while zeroing a single
term is still caught by ~100x.

**Cost:** the tolerance is loose in absolute terms. It is calibrated to catch
wrong kernels, not to certify bit-exactness, and the two are different goals.

### Decision: int8 quantization stays unconditional, including where no kernel can use it

**Why:** on arm64 the cascade is off, so the int8 codes are computed at insert
and never read — a wasted pass and 1 byte per dimension, about 25% of the
vector footprint.

Kept anyway, because **the stored format must not depend on the host's CPU
features.** A snapshot written on a machine without AVX2 would otherwise differ
in shape from one written on a machine with it, and a snapshot has to be
portable across machines — that is most of what it is for. Making the on-disk
and in-memory layout a function of the CPU that happened to write it turns a
portable artifact into a machine-specific one, which is a far worse property
for a database than 25% of vector memory.

It also means switching the cascade on, once a NEON kernel exists, needs no
reindex and no migration: the codes are already there.

**Cost:** ~25% of vector memory and one quantization pass per insert, unused on
any build without a SIMD int8 kernel.

### Decision: a NEON int8 kernel is a separate track, not part of this work

**Why:** the kernel that belongs in `kernel_arm64.go` is a NEON `SDOT`/`UDOT`
implementation. The instruction needs ARMv8.2-A dotprod, which the Ampere Altra
parts MinDB is deployed on do have, so this is real work rather than a
hypothetical — it is just a different kind of work from making the build
portable, and mixing them would mean shipping neither until both are done.

**Consequence, stated plainly:** until it lands, ARM deployments run the plain
float32 scan. Correct, and identical in results, but without the int8
bandwidth win — so ARM latency should be read against the brute-force column
of the benchmarks, not the cascade column.

---
## Durability

### Decision: `tmp → fsync → rename → fsync(parent dir)`, not a direct overwrite

**Why:** without the final directory fsync, the rename itself is not
guaranteed durable — a power loss right after `rename()` can leave the
directory entry pointing at the old file, or nowhere, even though the file's
own bytes are safely on disk. This is the step most homegrown atomic-write
implementations skip. (`pkg/core/snapshot.go:Save`)

**Windows caveat, documented rather than worked around:** Windows has no
directory-fsync equivalent, so `syncDir` is a no-op there
(`runtime.GOOS == "windows"`). Development happens on Windows; deployment
does not have to. Worth knowing rather than pretending the guarantee holds
everywhere.

### Decision: int8 codes are not persisted in the snapshot

**Why:** only the float32 vector is written to disk; codes, scales, and
residuals are recomputed by `store()` on load. Persisting them would grow
the file ~25% for a re-quantization cost (~300 ms at 100k×768) that's noise
next to reading hundreds of MB off disk — and, more importantly, it decouples
the on-disk format from the quantization scheme, so a future change to how
codes are built doesn't invalidate every snapshot already written.

### Decision: `Load` calls the internal `store()`, not the public `Insert()`

**Why:** `Insert` normalizes its input. A snapshot already contains
normalized vectors (they were normalized once, at the original insert); if
`Load` re-normalized them, float32 rounding means the norm computes as
`1 ± 1e-7` rather than exactly 1, and dividing by that shifts the vector's
low bits. Scores computed after a reload would then differ subtly from
scores computed before it — a snapshot round-trip should be invisible to a
caller, and this was the fix that made it so.

---

## Write-ahead log

### Decision: apply, then log, then fsync, then acknowledge — a redo log, not write-ahead ordering

**Why:** the ordering is Redis's AOF, not ARIES. The engine applies an
operation to the slab first and appends the record second
(`pkg/core/engine.go:Insert`), and the name "write-ahead log" is kept only
because that is what everyone calls the file.

The guarantee that matters is unchanged: **an operation is acknowledged only
after its record is durable, so every acked write survives a crash.** What
differs is the other side of it: an *unacked* write becomes visible to
readers before it is durable, so a `Search` running concurrently with an
in-flight `Insert` can return a vector that a crash then erases. No client
was ever told that `Insert` succeeded — from any client's point of view the
write was in flight, and an in-flight write may or may not be observed and
may or may not survive. A client that needs to know waits for the ack.

**What true write-ahead ordering would cost:** `store`'s only failure is
`ErrCapacityExceeded`, and it is discovered while applying. Logging first
means the record is already durable when the apply fails, so recovery would
replay an operation the engine rejected. Avoiding that means splitting the
allocator into reserve-then-fill so the failure is known before the append,
and it means a group-committed batch has to be applied in log order by one
writer rather than each writer applying its own — the batch stops being
independent work. That is a different write path, not a reordering of this
one.

### Decision: group commit with a leader, and no timer

**Why:** the writer that arrives while no sync is in flight becomes the
leader: it flushes everything buffered so far, records how far that reaches,
and runs the `fsync` with the lock released so later writers keep appending.
They land in the next batch. Nothing is timer-driven, so a lone writer syncs
immediately and pays no added latency, while under load the batch is however
many writers arrived during the last `fsync` — the system self-tunes to the
disk it is on.

**What it bought** (`BenchmarkWALInsert`, 13th Gen i7 laptop NVMe, Windows):

| writers | no log | fsync per write | group commit |
|---------|--------|-----------------|--------------|
| 1       | 425 ns | 288 µs          | 283 µs       |
| 8       | 501 ns | 301 µs          | 60 µs        |
| 64      | 638 ns | 229 µs          | 11 µs        |

At one writer the last two columns are the same benchmark, which is the
point: the batch only exists when somebody else is waiting. The absolute
numbers are a property of the disk — expect a different answer on a cloud
volume with a network in the way — but the shape holds.

**What it cost:** durability is ~700x the cost of the in-memory write at one
writer. That is the price of the guarantee, not an implementation artifact,
and it is why `-wal off` exists.

### Decision: the record checksum covers the length field, not just the body

**Why:** the length is read before the body and decides how many bytes are
read. A checksum over the body alone validates whatever the corrupt length
said to read, so a flipped bit in the length field either walks off the end
of the file or silently swallows the next record. Covering both means a
damaged length is caught by the same check as damaged data.

### Decision: the log stores normalized vectors, and replay calls `store()`

**Why:** the same reason `Load` does (see Durability). `Insert` normalizes;
re-normalizing an already-unit vector is not the identity in float32, so a
replay that went through `Insert` would shift low bits and make scores after
a recovery differ from scores before the crash. The log therefore holds what
the slab holds, and replay writes it straight back.

### Decision: `Save` rotates the log first, then writes the snapshot, then the meta, then deletes the retired segment

**Why:** each step is load-bearing, and the order is the only one that
survives a crash at every point:

- **Rotate first** so writes landing during the snapshot go to the new
  segment. Deleting the old segment afterwards then cannot lose them.
- **Meta before delete, never after.** Recovery accepts a log when some
  retained segment carries the meta's run id. With the delete first, a crash
  in between leaves a meta naming a run id no remaining segment has, and the
  engine refuses to start on a set of files that is perfectly consistent.

A crash anywhere in the sequence leaves the log covering *more* history than
the snapshot needs, never less. Replaying the overlap is harmless: insert and
delete are blind writes with no read-modify-write, so re-applying a range
that the snapshot already contains converges on the same state.

### Decision: snapshot lineage lives in a companion `<snapshot>.meta` file, not in the snapshot format

**Why:** the failure being prevented is an operator restoring last week's
snapshot next to this morning's log, and having the log replayed over a
history it never belonged to. Every completed `Save` mints a run id, stamps
it into the segment it rotates to, and writes it beside the snapshot;
recovery refuses to start unless some retained segment carries the id the
meta names, with an error that says how to fix it — replace the snapshot,
its meta and the log together, or delete the log so the snapshot stands
alone.

**Rejected alternative: a snapshot format version bump with the run id
inside.** It works, but it makes every existing snapshot unreadable to get a
field that has nothing to do with the snapshot's contents. A sidecar file
costs one `os.Stat` and keeps v1 snapshots loadable.

**What it cost:** a snapshot with no meta beside it — one written before
this existed, or one restored without its companion — cannot be checked at
all. That case is accepted, replayed, and reported (`WALRecovery.LegacyMeta`,
logged as a warning at startup) rather than refused, because refusing would
break every engine that predates the file.

### Decision: a failed fsync marks the log unhealthy forever, and the process keeps serving

**Why:** halting on a write error throws away a correct in-memory engine that
can still answer every read. Pretending nothing happened is worse. So the
process stays up, `wal_healthy` goes false, and it stays false: the readiness
probe reports `NOT_SERVING`, the orchestrator takes the pod out of rotation,
and reads continue for whoever is already connected.

**Why it never recovers:** on Linux a failed `fsync` may already have dropped
the dirty pages it could not write. The data is gone, and the next call has
nothing left to fail on — so a retry returning success proves nothing about
the write that failed. For the same reason writes are refused from then on
rather than acknowledged on the strength of a later successful sync.

### Decision: damage ends the log, and that decision is made permanent

**Why:** a process killed mid-append leaves a partial record at the end of
the file. That is how a log is *expected* to end, not corruption, so replay
stops there and the open succeeds.

What matters is that the decision sticks: recovery truncates the damaged
segment to the last good record and deletes every segment after it. Without
that, the next restart would replay records this one discarded, and two
recoveries over the same files would disagree about what the database
contains.

A segment shorter than its header is the same case one step earlier — a
crash between creating the file and syncing its header. No record can exist
in it, so it is removed rather than treated as an unreadable log.

### Decision: `Delete` returns `(bool, error)`, and no-op deletes are not logged

**Why:** the bool is "did this id exist", which the RPC needs to report
`deleted_count`. The error is the durability failure, and swallowing it is
the one thing a delete must never do: an acknowledged delete that is lost
resurrects the vector on the next restart, which is worse than a failed
delete the caller can retry.

Deletes that change nothing are not logged at all. Logging them would let a
caller retrying a delete in a loop grow the log without ever changing the
state it describes.

### Decision: the log is on by default wherever snapshots are

**Why:** `-wal` defaults to `<snapshot>.wal`, and requires `-snapshot`
because nothing retires segments without snapshots — the log would grow
until the disk filled. A snapshot on its own silently loses every write since
it was taken, so the safe combination is the default one, and `-wal off` is
how to say that losing them is acceptable.

The startup log warns when `-snapshot-interval` is 0, because then nothing
retires a segment until shutdown and the log grows unbounded in the meantime.

### Decision: the health service listens on its own port

**Why:** the data server is built with `grpc.ForceServerCodec`, which applies
to every service registered on it, and the FlatBuffers codec type-asserts
what it is handed to `*flatbuffers.Builder`. A protobuf health response on
the same server would panic rather than fail cleanly. The alternative —
registering the codec by content-subtype instead of forcing it — would change
the contract every existing client is built against. A second listener on
`-health-addr` costs a port and nothing else.

---

## Point lookups and introspection

### Decision: `Get` omits ids it cannot find, rather than returning `NotFound` or a null placeholder

**Why:** the alternatives are a per-id error, which makes one unknown id in a
batch of a hundred fail the other ninety-nine, or a positional response with
holes in it, which makes every caller carry a null check for a case that is
routine rather than exceptional. A recommendation service asking for the
metadata of ten candidate ids, one of which was deleted a second ago, wants the
nine.

So: found records in request order, missing ids simply absent, no error. A
caller that needs to know which ids missed compares what came back against what
it asked for.

**Cost, stated so nobody trips over it:** `len(result) < len(ids)` is normal,
and **the result is not positionally aligned with the request**. Callers match
on id, not on index. Batch and single-id behave identically, which is the point
— there is no special case to get wrong.

### Decision: `Get` copies the vector and the payload out of the slab

**Why:** the slab is mutable and its slots are recycled through the free list.
Handing back a sub-slice of `e.vectors` would let a later `Insert` — possibly
under a *different* id that inherited the slot — rewrite a response the caller
is still holding, with no lock left to protect it. The resulting corruption
would be silent, non-deterministic, and attributed to anything but `Get`.

**Cost:** one allocation per hit, `dims*4` bytes plus the payload. That is the
price of a result that stays valid after the read lock drops, and it is not
negotiable at any read path that outlives the lock.

### Decision: `Get` returns the normalized vector; the original norm is not stored

**Why:** `Insert` normalizes to unit length and discards `|v|`, because
normalization is load-bearing twice — it makes cosine similarity equal the dot
product, and it is a precondition of the Cauchy-Schwarz bound. So `Get` returns
`v/|v|`, not `v`. For cosine similarity the two are interchangeable; the
magnitude is simply not recoverable.

Storing it would cost 4 bytes per vector and a snapshot format version. Nothing
has needed it, so it is documented rather than built.

**Worth being precise about what is exact:** the vector comes from the float32
slab, not from the int8 codes. The codes exist only to prune candidates during
search and are never a source of truth, so `Get` is not lossy in the way the
quantization might suggest — only in the way normalization is.

### Decision: `Stats.memory_bytes` is reserved, not used; `payload_bytes` is maintained incrementally

**Why:** allocation is eager, so "memory used" and "memory reserved" are the
same number the moment after `New` as they are at capacity. Reporting it as
usage would make a fresh engine look full; the field is named and documented as
the reservation it is.

`payload_bytes` is the one genuinely variable figure, payloads being the only
on-demand allocation the engine owns. It is kept as a running total updated by
every path that stores or drops a payload, because summing it on demand would
make `Stats` O(capacity) — an introspection call that gets slower the bigger
the deployment is, which is exactly backwards for something a monitor polls.

**Cost:** a counter that two call sites have to keep honest, and a test
(`TestStatsPayloadBytesTracksMutations`) whose job is to notice when a third
one forgets.

### Decision: no `segment_count` in `StatsResponse`

**Why:** MinDB is a flat slab. A field that always reports `1` does not describe
the system, it describes a system somebody expected to find. FlatBuffers can
append the field when segments actually exist, at no wire cost to existing
clients — which is precisely the property that makes waiting free.

### Decision: `flatc` is pinned in a container, its output is committed, and CI checks the two agree

**Why:** `flatc` and the FlatBuffers Go runtime are a matched pair, and a
mismatch **does not fail the build**. It shifts vtable offsets and surfaces as
garbage field values at runtime, in a service that looks healthy. Pinning
`flatc` to the `v25.12.19` that matches `go.mod`, in a container rather than on
`PATH`, makes the pair explicit.

Committing the output keeps a plain `go build` free of any toolchain but Go.
`make check-gen` regenerates and fails on a diff, which is the only thing
stopping the schema and the committed code from drifting apart unnoticed — and
it caught exactly that on its first run: the committed code predated several
gRPC API changes (`grpc.Invoke`, `*grpc.ClientConn` instead of
`grpc.ClientConnInterface`, no `UnimplementedVectorServiceServer`).

**Cost:** regeneration needs Docker, and a `.gitattributes` entry pinning
`pkg/mindb/*.go` to LF so that `autocrlf` on Windows does not make `check-gen`
report every line of every file as changed.

### Decision: `api.Server` embeds `mindb.UnimplementedVectorServiceServer`

**Why:** without it, adding an RPC to the schema breaks `api.Server` at compile
time until a handler exists. That sounds like the safer default, and it is the
reason the alternative needs stating: it also means a schema change cannot be
generated and reviewed separately from the handlers that implement it, which is
how the generated code drifted out of date in the first place.

**Cost, and it is a real one:** an RPC that is declared and never implemented
now returns `Unimplemented` at runtime instead of failing the build. The
mitigation is that every RPC has an API-level test that goes over a real
loopback gRPC connection, so an unimplemented handler fails a test rather than
a deployment.

---
## Wire protocol

### Decision: FlatBuffers over gRPC, with a genuinely zero-copy read path for vectors

**Why:** the generated FlatBuffers accessor reads one `float32` per call
through bounds-checked offset arithmetic — correct, but it copies element by
element. `pkg/api/grpc_server.go:float32Vector` instead validates the vector
region once and then reinterprets the wire buffer directly via
`unsafe.Slice`, which is safe here because FlatBuffers' wire format is
already little-endian 4-byte-aligned float32 — exactly what a `[]float32` on
a little-endian platform looks like in memory. A `nativeLittleEndian` runtime
check guards a byte-swapping fallback for the (currently theoretical, since
Go's amd64/arm64 targets are all little-endian) big-endian case, rather than
assuming it.

**Honesty about "zero-copy":** it's true for reading a request and false for
building a response — gRPC still marshals whatever the handler returns, and
constructing a `flatbuffers.Builder` for a `SearchResponse` still allocates.
The docs say which half is real rather than claiming the whole thing.

### Decision: `grpc.ForceServerCodec(flatbuffers.FlatbuffersCodec{})` is mandatory, not optional

**Why:** the RPC handlers return `*flatbuffers.Builder`, not a generated
protobuf struct, so gRPC's default codec cannot marshal the response — this
fails at **request time**, not at compile time, if the codec isn't
registered. `cmd/mindb-server/main.go` calls this out explicitly at the
`grpc.NewServer(...)` call site because it's the kind of thing that works in
every test that constructs the server directly and only breaks over the
wire.

### Decision: `Insert` batches are not transactional

**Why:** the engine has no transaction log or rollback mechanism, so if
vector 5 of a 10-vector `Insert` request fails validation, vectors 0–4 are
already stored and stay stored — the RPC just returns an error naming which
vector failed and how many were stored before it
(`pkg/api/grpc_server.go:insertError`). Pretending otherwise (silently
rolling back) would require machinery the engine doesn't have and that a
sidecar with this scale target doesn't need; documenting the real behavior
was cheaper and more honest than building it.

---

## Packaging

### Decision: the builder stage runs on the build platform and Go cross-compiles

**Why:** `--platform=$BUILDPLATFORM` on the first stage, `GOARCH=$TARGETARCH`
on the `go build`. The obvious alternative — let buildx run an arm64 builder
under QEMU and compile natively inside it — emulates the entire Go toolchain
to produce a binary Go can cross-compile in seconds. The arm64 leg goes from
minutes to about as long as the amd64 one.

This is only free because `CGO_ENABLED=0`. With cgo there would be a C
cross-toolchain to install and a sysroot to manage, and the emulated builder
starts to look reasonable. MinDB has no cgo in the server binary, so it is
free, and the same flag is what makes the static base below possible.

### Decision: `distroless/static-debian12:nonroot`, not Alpine or scratch

**Why:** the binary is static, so nothing in the image is needed at runtime
except CA certificates, timezone data and a passwd entry for a non-root uid.
That is exactly what distroless/static carries and roughly all it carries —
about 2 MB, no shell, no package manager, no libc to keep patched. Alpine
brings musl, busybox and apk for no benefit here and a standing patch
obligation. `scratch` would work too, but then the non-root uid has no passwd
entry and the certs have to be copied in by hand; distroless is that, already
assembled.

**What it cost:** there is no shell, so `kubectl exec` into a wedged pod gets
you nothing. That is the point — the attack surface is the same emptiness —
but it means debugging is done through the health service, the `Stats` RPC and
the logs. `:debug-nonroot` is the escape hatch on the day it matters, and it
is a one-word change.

### Decision: no `CMD`, no `VOLUME`, and `/data` as the working directory

**Why:** every flag is a deployment decision, and the one that matters most is
`-snapshot` — without it the server is memory-only and the write-ahead log is
off, which is a reasonable thing to want and a terrible thing to get by
accident. A default `CMD` would pick one of those for the operator silently.

`VOLUME /data` is worse than useless: under `docker run` it creates anonymous
volumes that outlive nothing in particular, and Kubernetes ignores it
entirely. A `WORKDIR` documents the same intent and changes no behaviour. What
does need saying out loud is that the mount has to be writable by uid 65532.

### Decision: arm64 is tested on a native runner, not under QEMU

**Why:** arm64 is the deploy target and it is not the same build. `HasFastInt8`
is false there, so the cascade never runs and the brute-force scan is what
actually serves every query — a path that on amd64 is only ever exercised as
the differential test's control. It deserves the same suite, not a subset.

GitHub's `ubuntu-24.04-arm` runners are free for public repositories, which
removes the reason QEMU was the plan. Under emulation the suite is roughly an
order of magnitude slower, and `go test -race` is worse than slow: the race
detector's timing assumptions do not survive instruction emulation, so a
green run proves less than it appears to. A native runner gives a real
`-race` on the architecture that will actually run in production.

QEMU is still registered in the image job — but only to assemble a manifest
list, never to compile.

### Decision: the image is built on every push and pushed only on a tag

**Why:** a Dockerfile that is only exercised at release time breaks at release
time. Building it on every push and every pull request costs a few minutes of
CI and moves that failure to the change that caused it. Publishing, by
contrast, wants to be deliberate: GHCR gets a push only for `v*` tags, and
`docker/metadata-action` derives `vX.Y.Z`, `vX.Y`, `latest` and a long SHA tag
from the ref.

The image name comes from `github.repository` rather than a literal, so a
repository transfer needs no edit here — which is not hypothetical, this
repository has had one.

### Decision: the version is stamped at link time and logged first

**Why:** `-X main.version`, defaulting to `dev`. A binary running on a machine
nobody has a terminal for should be able to say what it is, and the first line
of the startup log is where that belongs. The default is deliberately not a
fake version number: `dev` means "this came out of somebody's working tree and
its git state is unknown", which is true and useful, where `v0.0.0` would be
a claim.

### Decision: benchmarks are reported per architecture, each labelled with its machine

**Why:** a single benchmark table implies a single number, and there isn't
one. On amd64 with AVX2 the cascade is live and beats brute force; on arm64
`HasFastInt8` is false, the cascade is switched off, and the same query runs
the float32 scan. Averaging those or quoting whichever is flattering would
describe a deployment that does not exist.

So `docs/README.md` carries two tables, each naming the machine it was
measured on, and `make bench` prints `GOARCH` before the numbers so a pasted
result carries its own provenance.

**Numbers from emulation are not published.** A benchmark run under QEMU
times the emulator. The arm64 figures come from a native runner via a
hand-triggered workflow, and where no native measurement exists the table says
so rather than borrowing one.

### Decision: the module path moved to the new owner before the first tag

**Why:** the repository was transferred and `go.mod` still declared
`github.com/typicallhavok/mindb`. GitHub redirects a clone, so leaving it alone
would have worked — `go get` on the old path resolves, because the redirect
lands on a repository whose `go.mod` names that same old path. What would not
work is `go get` on the *new* path, which Go rejects outright when the module
path inside `go.mod` disagrees with the path asked for.

So the choice was: an honest path that breaks anyone already on the old one, or
a working path that permanently advertises an account that no longer owns the
code. A module path is only cheap to rename while there are no consumers, and
there were none — the first tag had not been cut. Renaming first means
v0.1.0 is the first thing anyone can pin and it names the right owner.

The old path stops resolving from that commit on. Nothing else needed editing:
the image name comes from `github.repository`, not a literal.

---

## What was tried and explicitly rejected

Kept here so nobody re-proposes them without knowing why they didn't work.

- **AVX-512 on the float32 path.** A plain unrolled-by-8 Go loop already
  sustains ~13 GB/s per core against an ~11.7 GB/s memory subsystem — there
  is nothing left for SIMD to win when the loop is already waiting on DRAM.
  Assembly only pays for itself on the int8 path, where pure Go is
  compute-bound rather than memory-bound.
- **4-bit quantization.** Its error bound is technically valid, but wide
  enough that it prunes almost nothing — the 8x compression bought a full
  scan followed by a full rescore, which is strictly worse than just
  scanning float32 once. Measured, then deleted.
- **Plain sign-bit (1-bit) quantization.** The fastest possible scan
  (~1 ms for 100k vectors) but recall collapses on clustered data
  (`recall@10 ≈ 0.63` even with a 1000-candidate shortlist), because
  vectors that share a cluster also share most of their sign bits — Hamming
  distance can't discriminate exactly where the top-k lives. Deferred to a
  future rotated-quantization (RaBitQ-style) approach rather than shipped
  broken; tracked as stage 4 in the roadmap.
