# MinDB Architecture

This is the design record for MinDB v2. It documents what the system does, why it does it
that way, and — deliberately — the approaches that were measured and rejected. If you only
read one section, read [The thesis](#the-thesis).

Numbers marked **(measured)** come from benchmarks on the development machine and are
reproducible with `go test -bench`. Numbers marked **(projected)** are arithmetic from the
bandwidth model and have not been observed on that hardware.

---

## The thesis

> Latency is bytes-moved ÷ bandwidth.
> The entire design is about moving fewer bytes **without giving up correctness.**

An exact k-NN scan over `N` vectors of `dims` float32 must stream `N × dims × 4` bytes.
At 100,000 × 768 that is 293 MB. No amount of kernel tuning beats the memory system: if
your machine sustains 11.7 GB/s, the scan cannot finish faster than ~25 ms. This is
physics, not an implementation detail, and it is the constraint every decision below
answers to.

There are only two ways to go faster: move fewer bytes, or move them from faster memory.
MinDB does the first, and does it in a way that provably does not change the answer.

---

## Bound-and-refine

Every mainstream vector database that offers quantization — Qdrant, Weaviate, Milvus —
uses it *heuristically*. Score with the compressed codes, oversample by some constant,
rescore the shortlist, and hope the true neighbours were in it. Recall is empirical and
unbounded: you find out it was 0.97 by measuring, and there is no guarantee for any
individual query.

MinDB uses quantization **only as a filter, never as an answer.** The filter is provably
lossless, so the result is bit-identical to brute force.

### The bound

Vectors are unit-normalized at insert, so cosine similarity **is** the dot product. Each
vector `v` is stored twice — exact float32, and an int8 code whose reconstruction is `v̂`.
One extra float32 per vector holds the residual norm `ρ_v = ‖v − v̂‖`.

Scoring is **asymmetric**: the query stays in float32 and is never quantized. That
collapses the error to a single term, which Cauchy–Schwarz bounds exactly:

```
s  =  q·v  =  q·v̂ + q·r_v            where  r_v = v − v̂

|q·r_v|  ≤  ‖q‖·ρ_v  =  ρ_v          since  ‖q‖ = 1

    lo_i = q·v̂ᵢ − ρᵢ          hi_i = q·v̂ᵢ + ρᵢ
```

Every vector therefore yields an interval `[lo_i, hi_i]` that is *guaranteed* to contain
its true score.

### The algorithm

1. Scan all `N` vectors in int8. Compute `lo_i` and `hi_i`.
2. Let `τ` be the k-th largest `lo_i`.
3. Any vector with `hi_i < τ` **cannot** be in the true top-k. Discard it. This is a
   proof, not a heuristic.
4. Rescore the survivors in exact float32.

**(measured)** at N=50,000, 768 dims, clustered data: 39 survivors out of 50,000 —
**0.078%** — with the top-k always retained, because it cannot be otherwise.

### Why asymmetric, and one correction worth recording

An earlier draft of this design used symmetric scoring (quantize the query too) and
derived `‖v̂‖ = √(1 − ρ²)`. That derivation is **wrong**: it assumes the residual is
orthogonal to the reconstruction, and int8 rounding is not an orthogonal projection.
Measured deviation was 0.000224 — small, but "approximately" voids a proof, and the
exactness guarantee is the entire point of this design.

Asymmetric scoring removes the `‖v̂‖` term from the bound altogether, so the unsound step
is gone rather than patched. `ρ_v` is computed directly at insert with no assumptions,
and is stored, never derived.

It is also simply better on every axis:

| codes | mode | survivors | pct | max ε | bound valid |
|---|---|---|---|---|---|
| int8 (4x) | symmetric | 168 | 0.335% | 0.02109 | ✅ |
| int8 (4x) | **asymmetric** | **39** | **0.078%** | **0.01249** | ✅ |
| 4-bit (8x) | symmetric | 50000 | 100.000% | 0.46844 | ✅ |
| 4-bit (8x) | asymmetric | 49994 | 99.987% | 0.23511 | ✅ |

**(measured)**, N=50k, 768 dims, clustered synthetic data. Uniform-random data would have
flattered these numbers considerably, so clustered data is used throughout.

Asymmetric is 4.3x tighter, **and** it needs only AVX2 + FMA3 rather than AVX-512 VNNI,
because the query side stays in float32.

**4-bit is dead.** Its bound is valid but useless — it prunes essentially nothing, so 8x
compression buys you a full scan followed by a full rescore. It is not in the codebase.

### The adaptive guard

If survivors exceed a threshold fraction of `N`, the rescore is skipped and a plain
float32 scan runs instead, so the worst case is brute force rather than worse than it.

Break-even on sequential bytes alone is ~75% (`73 + f×293 = 293`), but rescoring survivors
is *random* access and less efficient per byte, so the practical crossover is lower. It
ships as a tunable constant.

---

## Why the assembly lives at the int8 tier

**(measured)** cost of one full scan, per representation, N=100,000 at 768 dims, single
core:

| tier | bytes/vector | scan size | time | rate |
|---|---|---|---|---|
| float32 (exact) | 3072 | 293 MB | **50.0 ms** | 5.7 GB/s |
| int8 (pure Go) | 768 | 73 MB | **103.2 ms** | 0.7 GB/s |
| 1-bit (POPCNT) | 96 | 9.2 MB | **1.0 ms** | 8.9 GB/s |

The int8 row is the whole story: it moves **4x less data than float32 and takes twice as
long.** It is compute-bound, not memory-bound. And not because the loop is badly written —
four shapes were tried:

```
unroll4          98.19 ms   0.78 G MAC/s
unroll8         102.11 ms   0.75 G MAC/s
unroll16         93.08 ms   0.83 G MAC/s
word-at-a-time  168.15 ms   0.46 G MAC/s
   needed to be DRAM-bound:  6.0 G MAC/s
```

Pure Go plateaus at **0.83 G MAC/s, 7.2x short of the memory wall.** This is not a tuning
problem. `VPMADDUBSW`+`VPMADDWD` performs 32 int8 multiply-accumulates per instruction on
AVX2; `VPDPBUSD` does 64 on AVX-512 VNNI. Go's compiler emits neither and offers no
intrinsics.

So assembly is the only route to this tier, and **without it the tier is worse than
useless** — the cascade would be slower than the brute force it replaces.

This is worth stating plainly, because it inverts the usual justification. On the float32
path, hand-written SIMD is decoration: the pure-Go unroll-8 kernel already sustains
13.0 GB/s per core **(measured)** against a memory subsystem that delivers 11.7 GB/s
across all cores. There is nothing for SIMD to win. On the int8 path the assembly is
load-bearing — it is the component that makes the architecture net-positive.

**Hand-tuned SIMD is not the novelty here.** Faiss, hnswlib, usearch and Milvus all ship
it; it is table stakes. The novelty is the provable pruning above. The assembly is what
makes that pruning pay for itself.

---

## What does not work

Documenting this is not a weakness. It is the part that shows the numbers were actually
measured rather than assumed.

### The 1-bit tier fails on recall

By far the fastest scan — 9.2 MB, 1.0 ms for all 100k **(measured)** — and the only
genuine route to sub-millisecond search at this scale. Recall with plain sign-bit
quantization, however:

```
shortlist   100: recall@10 = 0.220
shortlist   300: recall@10 = 0.407
shortlist  1000: recall@10 = 0.633
```

0.63 is not shippable, and the cause is structural rather than fixable by tuning:
clustered vectors share most of their sign bits, so Hamming distance cannot discriminate
*inside* a cluster — which is exactly where the top-k lives.

The known fix is a random rotation (a fast Hadamard transform) before binarizing, which
spreads information across all dimensions. This is the core idea of RaBitQ (SIGMOD 2024)
and would plausibly reach 0.95+.

It is **future work, explicitly research-risk**, deferred to different hardware and its
own branch. Nothing in the current design depends on it, and no extension points have been
built for it — speculative seams for a feature that may never ship are exactly the
complexity this design avoids.

### Sub-millisecond at 100k × 768 is not claimed on consumer hardware

73 MB has to move. Where it *does* become true **(projected, from `bytes ÷ bandwidth`)**:

| memory bandwidth | cascade (73 MB) | brute force (293 MB) |
|---|---|---|
| 11.7 GB/s (dev laptop, 4 threads) | 6.1 ms | 24.5 ms |
| 45 GB/s (mid server) | 1.6 ms | 6.4 ms |
| 100 GB/s | **0.71 ms** | 2.9 ms |
| 200 GB/s | **0.36 ms** | 1.4 ms |

Sub-millisecond *exact* k-NN at 100k × 768 is reachable on server-class memory — and only
the cascade gets there.

---

## Concurrency: lock-free was evaluated and rejected

The original design claimed lockless concurrency via tombstoning. That is a category
error. Tombstoning solves *deletion without shifting memory*; it says nothing whatsoever
about whether concurrent readers and writers are safe. Concretely, the design had:

- **A hard crash.** `idMap` is a plain Go map. Two concurrent `Insert`s, or an `Insert`
  racing a `Delete`, hit the runtime's `concurrent map read and map write` **throw** —
  unrecoverable, not a `panic`, cannot be deferred or recovered. The process dies.
- **Silent corruption.** With no release store on publication and no acquire load on the
  reader, a searcher can score a half-written vector. It does not error; it returns a
  plausible *wrong* score. This is the worst failure mode a database can have.
- **Lost writes.** RCU compaction builds a copy while concurrent writes land in the *old*
  engine and vanish at the pointer swap. Real RCU is "lock-free **readers**, *synchronized*
  writers" — it never claimed to remove writer locking.
- **A subsystem that reclaimed nothing.** Capacity is preallocated, so the backing array
  is the same size before and after compaction. The background goroutine, the second
  engine copy, the pointer swap and the transient 2x memory spike all existed to solve a
  problem a free list solves in ten lines.
- **Torn snapshots**, and slot reuse that could pair vector A's score with vector B's ID.

### The arithmetic

`RLock`/`RUnlock` costs ~20 ns. A cascade search costs ~6,000,000 ns.

**The lock is 0.0003% of the operation.** The lockless design was trading a crash bug, a
silent-corruption bug and a lost-write bug for a rounding error.

### The resolution

One `sync.RWMutex`, RLock held across the whole scan, plus a free list replacing
compaction outright. Every failure above is gone by construction. Concurrent searches
still run fully in parallel, because RLock is shared.

The one real cost is that a writer waits behind an in-flight scan. For a read-dominated
sidecar that is the correct trade — and it is *documented* rather than pretended away.

---

## Layout

```go
type Engine struct {
    dims, capacity int

    mu         sync.RWMutex
    vectors    []float32   // capacity*dims, unit-normalized at insert
    codes      []int8      // capacity*dims, int8 quantization of the above
    scales     []float32   // capacity, per-vector quantization scale
    residuals  []float32   // capacity, ρ = ‖v − v̂‖ — the bound, stored not derived
    externalID []string
    payloads   [][]byte
    live       []bool
    idMap      map[string]uint32
    free       []uint32    // recycled slots; replaces compaction entirely
    highWater  uint32      // slots beyond this were never used; scan stops here
}
```

Struct-of-arrays, so a scan touches only the arrays it needs and every cache line it pulls
in is entirely useful.

**Normalization is load-bearing twice.** It makes cosine similarity equal the dot product,
which removes a division and a random-access magnitude load from the hot loop and deletes
an entire array. It is also a precondition of the Cauchy–Schwarz bound, which assumes
`‖q‖ = ‖v‖ = 1`.

**Top-k uses a bounded min-heap of size k**, not a sort. Sorting 100k scores is O(n log n)
plus allocation and would erase every gain above. One per worker, merged at the end,
pooled across queries. `if score <= heap.min { continue }` is a single predictable branch.

**Memory:** `capacity × dims × 5 bytes` plus payloads. At 768 dims that is 3072 B float32 +
768 B codes + 8 B metadata = 3848 B/vector, so **≈368 MiB at 100k × 768**, allocated
eagerly at boot. Eager allocation means the process either has the memory or fails
immediately, rather than failing under load.

**This is a flat slab, and it is worth naming as one.** There are no segments, no write
buffer and no tombstones: one set of parallel arrays, an id map from external ID to slot,
a free list that recycles deleted slots, and whole-file snapshots. That is what makes a
scan sequential and deletes compaction entirely, and it is also why every write takes the
same lock every reader uses, and why nothing survives a crash between snapshots. A
write-ahead log and then a segmented store are the answers to those two, in that order,
and both are separate tracks with their own specs — see the roadmap. Until they land,
read this section as the whole of MinDB's storage layer, because it is.

---

## Durability

Snapshots are written `tmp → fsync → rename → fsync(parent directory)`.

The parent-directory fsync is the step almost everyone omits, and without it the *rename
itself* can be lost in a crash — you fsync the data, rename over the old file, lose power,
and come back to a directory entry that was never durably updated. The file format is a
header (magic, version, dims, count) plus a CRC32, so a truncated or corrupted snapshot is
rejected at load rather than silently loaded as garbage.

Two platform caveats, since development happens on Windows: there is no directory fsync,
and `os.Rename` cannot replace a file that is currently open.

**The snapshot stores float32 vectors, IDs and payloads only. int8 codes are recomputed at
load.** Persisting them would grow the file ~25% (366 MB vs 293 MB at 100k × 768), and
re-quantizing costs ~300 ms — noise next to reading 293 MB off disk. More importantly it
decouples the on-disk format from the quantization scheme, so changing how codes are built
does not invalidate every existing snapshot in the field.

---

## Wire protocol

gRPC with FlatBuffers, defined in `fbs/mindb.fbs`: Insert, Search, Delete, Snapshot, Get,
Stats. Fields are only ever appended, never renumbered or removed, so an older client keeps
working against a newer server.

Three things worth knowing:

**"Zero-copy" is half true, and it is worth being precise about which half.** Reading a
request genuinely is zero-copy — the buffer arrives little-endian and 4-byte aligned, so
the query vector is reinterpreted in place via `unsafe.Slice` rather than read through the
generated per-element accessor, which is bounds-checked offset arithmetic per element.
*Building* a response, however, allocates. FlatBuffers does not change that.

**The generated handlers return `*flatbuffers.Builder`**, which requires
`grpc.ForceServerCodec(flatbuffers.FlatbuffersCodec{})` at server construction. Omit it and
the server fails at *runtime*, not compile time.

**`Get` returns what was stored, which is not what was sent.** `Insert` normalizes and
discards the original magnitude, so a point lookup returns `v/|v|`. The values are exact
— read from the float32 slab, never reconstructed from the int8 codes — and ids that
cannot be found are omitted from the response rather than erroring, so the response is not
positionally aligned with the request.

`pkg/mindb/` is flatc-generated. Do not hand-edit it; `make gen` overwrites it, using a
pinned `flatc` container that matches the FlatBuffers runtime in `go.mod`. `make check-gen`
fails when the committed output and the schema have drifted apart.

---

## Claims explicitly retired

The previous design asserted several things that do not survive contact with measurement.
They are listed here so nobody reintroduces them.

- **"Bypasses the Go garbage collector entirely."** `sync.Pool` does not bypass the GC —
  pooled objects are dropped on GC cycles. It lowers allocation *rate*, which is
  worthwhile and is what should be claimed.
- **"Zero heap allocations on the query path."** Unachievable end-to-end: gRPC allocates
  per RPC and building the FlatBuffers response allocates. The defensible version —
  **zero allocations in the scan kernel** — is true and worth stating.
- **"Sub-millisecond at 100,000 vectors."** Off by 25–50x at 768 dims on consumer
  hardware. See the bandwidth table above for where it becomes true.
- **"`uint32` addressing is ultra-fast."** Go slice indices are `int`, so `uint32` adds a
  conversion. The real benefit is smaller `idMap` and free-list entries. Keep the type,
  fix the reasoning.

---

## Requirements

**Hard floor:** Go 1.21+, any 64-bit Go platform. RAM per the layout section above.

**For the cascade speedup:** AVX2 + FMA3 — Intel Haswell (2013)+ or AMD Excavator (2015) /
Zen (2017)+. **AVX-512 and VNNI are not required.** Without AVX2 the engine runs correctly
on a pure-Go fallback and warns loudly at startup, so nobody benchmarks the slow path by
accident.

---

## Roadmap

| stage | contents | state |
|---|---|---|
| 1 | correct exact engine — `pkg/math`, `pkg/core`, `pkg/api`, server | done |
| 2 | bound-and-refine cascade in pure Go, differential test | done |
| 3 | Avo-generated AVX2 asymmetric int8 kernel | done |
| 4 | rotated 1-bit tier (RaBitQ-style) | deferred, research-risk |
| — | portability: kernels split by build tag, arm64 on the pure-Go path | done |
| — | `Get` and `Stats` RPCs | done |
| — | NEON `SDOT` int8 kernel for arm64 | planned |
| — | write-ahead log: replay onto the slab at boot, truncate on snapshot | planned |
| — | segmented store: write buffer, immutable segments, tombstones | planned |

Each stage ships something working. Stage 2 is expected to be *slower* than stage 1 — its
job is to establish correctness before any assembly exists to blame for a wrong answer.

---

## Verification

The differential test is the centerpiece: **the cascade must return exactly what brute
force returns**, across thousands of random queries and every kernel path, asserted on
both IDs and scores. A bound-validity test asserts `lo_i ≤ true_score_i ≤ hi_i` for every
vector; if that ever fails, the exactness guarantee is void and the project's central
claim is false.

Alongside it: `go test -race` including a test hammering concurrent Insert/Delete/Search,
snapshot round-trip plus a flipped-byte file rejected by checksum, and benchmarks
reproducing the tier tables above.

Pruning-ratio claims are validated against real data — `glove-100-angular` and
`gist-960-euclidean` (normalized) — via `make validate`, which brackets the dimension
range around the 768-dim target. Synthetic clustered data is a reasonable proxy but it is
not evidence.
