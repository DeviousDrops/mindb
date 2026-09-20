# MinDB

> An embedded, in-memory vector database that uses quantization as a **filter**, never as
> an answer — so search is 4x faster than brute force and the results are bit-identical to
> it.

MinDB is an exact k-NN vector engine for Go, built to run as a high-performance gRPC
sidecar. It caps its scale on purpose and spends the headroom on being provably correct
and cache-sympathetic rather than on an approximate index.

**Status: under construction.** See the [roadmap](#roadmap). The design is settled and
documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## The one idea

Exact k-NN is a memory-bandwidth problem, not a compute problem:

```
latency  ≈  bytes_scanned ÷ memory_bandwidth
```

Scanning 100,000 × 768 float32 vectors means moving 293 MB. On a machine sustaining
11.7 GB/s that is a **~25 ms floor**, and no kernel tuning gets under it.

So MinDB moves fewer bytes. It keeps an int8 copy of every vector (4x smaller) plus one
float32 per vector: the residual norm `ρ = ‖v − v̂‖` between the true vector and its
quantized reconstruction. Because vectors are unit-normalized and the query is never
quantized, Cauchy–Schwarz turns that residual into a **hard** two-sided bound on the true
score:

```
lo_i = q·v̂ᵢ − ρᵢ        ≤    true score    ≤        hi_i = q·v̂ᵢ + ρᵢ
```

Scan everything in int8, take `τ` = the k-th largest `lo_i`, and discard every vector with
`hi_i < τ`. Those vectors **cannot** be in the top-k — that is a proof, not a heuristic.
Rescore whatever survives in exact float32.

**Measured: 0.078% of the database survives.** The other 99.9% is eliminated with a
guarantee, not a hope.

### Why that's different

Qdrant, Weaviate and Milvus all offer quantization, and all use it *heuristically* — score
with compressed codes, oversample by a constant, rescore, hope the true neighbours were in
the shortlist. You learn your recall was 0.97 by measuring it, and no individual query
carries a guarantee.

MinDB's contract is different: **ANN-class pruning with exact-kNN guarantees.** The output
is bit-identical to brute force, and there is a differential test in CI asserting exactly
that across thousands of random queries.

---

## Numbers

MinDB runs two different search paths depending on the machine, so there is no single
number to quote. On x86-64 with AVX2 the bound-and-refine cascade is live. On arm64
`HasFastInt8()` is false, the cascade is switched off, and every query runs the
brute-force float32 scan. Both return identical results — the cascade is an
optimization, not an approximation — so what changes is latency, and it changes by
machine. Every table below names the machine it came from.

### Cost of one full scan, single core

Intel i7-5500U (2 cores / 4 threads), N=100,000 at 768 dims, **clustered** synthetic
data — uniform-random data would flatter these considerably.

| representation | bytes/vector | scan size | time |
|---|---|---|---|
| float32 (brute force) | 3072 | 293 MB | 50.0 ms |
| int8, pure Go | 768 | 73 MB | 103.2 ms |
| int8, AVX2 | 768 | 73 MB | above the memory wall (23.5 GB/s/core) |
| 1-bit (POPCNT) | 96 | 9.2 MB | 1.0 ms |

Note the int8 row: **4x less data, twice the time.** Pure Go cannot express the
instruction that makes int8 fast, which is why this project has assembly — and why the
assembly lives on the int8 path rather than the float32 one.

### End-to-end search, by architecture

`BenchmarkSearchPaths`, N=20,000 at 768 dims, k=10, clustered. The cascade is *forced on*
in both rows of each machine, which is the only way to see what the `HasFastInt8()` guard
is actually deciding.

| machine | kernel | brute force | cascade | cascade vs. brute force |
|---|---|---|---|---|
| Intel i7-5500U, 2c/4t (dev laptop) | `avx2` | 6.47 ms | **1.75 ms** | 3.7x faster |
| Intel Xeon Platinum 8573C, 4 vCPU (GitHub `ubuntu-latest`) | `avx2` | 2.41 ms | **1.16 ms** | 2.1x faster |
| Arm Neoverse-N2, 4 vCPU (GitHub `ubuntu-24.04-arm`) | `pure-go` | **1.42 ms** | 1.48 ms | 4% *slower* |

The ARM row is the whole argument for the guard, measured rather than assumed: with no
NEON int8 kernel, the cascade's extra pass costs more than the pruning saves, so
`HasFastInt8()` returns false and the engine takes the brute-force path. The server says
which one is live at startup: `kernel: name=pure-go fast_int8=false goarch=arm64`. See
[`DECISIONS.md`](../DECISIONS.md) under "Portability".

### The dot kernels, by architecture

`BenchmarkDot` and `BenchmarkDotInt8` at 768 dims, same two runners.

| machine | float32 | int8 |
|---|---|---|
| Xeon Platinum 8573C (AVX2) | 231 ns, 26.6 GB/s | **164 ns, 23.4 GB/s** |
| Neoverse-N2 (pure Go) | 243 ns, 25.3 GB/s | 268 ns, 14.3 GB/s |

Same code, opposite conclusion. With AVX2 the int8 kernel is the faster of the two; in
pure Go it is the slower one despite touching a quarter of the bytes, which is the
portability problem in one line.

### What durability costs

`BenchmarkWALInsert`, time per acknowledged insert. "off" is no log at all, the ceiling.
"per write" gives every writer its own `fsync`. "group commit" is the shipped path, where
one writer flushes and syncs for everyone who arrived while the last sync was in flight.

| machine | writers | off | per write | group commit |
|---|---|---|---|---|
| Xeon 8573C | 1 | 473 ns | 563 µs | 455 µs |
| Xeon 8573C | 8 | 623 ns | 451 µs | **108 µs** |
| Xeon 8573C | 64 | 626 ns | 418 µs | **17.1 µs** |
| Neoverse-N2 | 1 | 505 ns | 754 µs | 222 µs |
| Neoverse-N2 | 8 | 783 ns | 247 µs | **58.3 µs** |
| Neoverse-N2 | 64 | 839 ns | 195 µs | **12.0 µs** |

At one writer the last two columns are the same benchmark, which is the point: the batch
only exists when somebody else is waiting. Both runners have network-backed storage, so
read the shape and not the stopwatch — the absolute `fsync` cost is a property of the
disk, and the single-writer rows in particular move around between runs. The ratio at 64
writers is what transfers.

### Where sub-millisecond becomes real

Projected from `bytes ÷ bandwidth`, N=100,000 at 768 dims:

| memory bandwidth | MinDB cascade | brute force |
|---|---|---|
| 11.7 GB/s (dev laptop) | 6.1 ms | 24.5 ms |
| 45 GB/s (mid server) | 1.6 ms | 6.4 ms |
| 100 GB/s | **0.71 ms** | 2.9 ms |
| 200 GB/s | **0.36 ms** | 1.4 ms |

Sub-millisecond *exact* k-NN at 100k × 768 needs server-class memory — and only the
cascade gets there.

Reproduce any of the above with `make bench`, which prints `GOARCH` before the numbers.
The CI rows come from the `bench` workflow, run by hand on both runners; nothing here is
measured under emulation, because that would time QEMU rather than MinDB.

---

## Why there is assembly, and why it's on the int8 path

Hand-tuned SIMD is not a differentiator. Faiss, hnswlib, usearch and Milvus all ship it.
What matters is *where* it goes.

On the **float32** path it is decoration. A plain unrolled-by-8 Go loop already sustains
13.0 GB/s per core against a memory subsystem delivering 11.7 GB/s across all cores. There
is nothing for SIMD to win against a loop that is waiting on DRAM.

On the **int8** path it is load-bearing. Pure Go plateaus at **0.83 G MAC/s** — 7.2x short
of the memory wall — across four different loop shapes:

```
unroll4          0.78 G MAC/s
unroll8          0.75 G MAC/s
unroll16         0.83 G MAC/s
word-at-a-time   0.46 G MAC/s
   needed:       6.0 G MAC/s
```

That is not a tuning problem. `VPMADDUBSW`+`VPMADDWD` does 32 int8 multiply-accumulates in
one instruction; Go's compiler cannot emit it and offers no intrinsics. **Without the
assembly, the int8 tier is slower than not bothering** — the cascade would lose to the
brute force it replaces.

Target ISA is **AVX2 + FMA3**, not AVX-512. Keeping the query in float32 means the kernel
needs only `VPMOVSXBD` / `VCVTDQ2PS` / `VFMADD231PS`, which every x86 CPU since 2013 has.
The `.s` file is generated by [Avo](https://github.com/mmcloughlin/avo) and committed.

---

## Concurrency: lock-free was evaluated and rejected

MinDB uses a single `sync.RWMutex`. This is a deliberate reversal of an earlier lockless
design, and the reasoning matters more than the outcome.

`RLock`/`RUnlock` costs **~20 ns**. A search costs **~6,000,000 ns**. The lock is
**0.0003% of the operation** — and the lockless version was paying for that rounding error
with three real bugs:

- **A hard crash.** A plain Go map under concurrent access triggers the runtime's
  `concurrent map read and map write` *throw* — unrecoverable, not a `panic`. The process
  dies.
- **Silent corruption.** No publication barrier meant a searcher could score a half-written
  vector and return a plausible *wrong* score. No error, no signal.
- **Lost writes.** RCU compaction dropped writes landing in the old engine during the copy.
  RCU is "lock-free **readers**, *synchronized* writers"; it never claimed otherwise.

Compaction also reclaimed nothing, since capacity is preallocated — an entire background
subsystem solving a problem a ten-line free list solves with no concurrency exposure.

Concurrent searches still run fully parallel, because RLock is shared. The one real cost —
a writer waits behind an in-flight scan — is the right trade for a read-dominated sidecar,
and it's documented rather than pretended away.

Full detail in [`ARCHITECTURE.md`](./ARCHITECTURE.md#concurrency-lock-free-was-evaluated-and-rejected).

---

## What doesn't work

**The 1-bit tier fails on recall**, and it's in this README because it was the most
promising idea here. It's the fastest scan by far — 1.0 ms for all 100k, the only genuine
route to sub-millisecond on consumer hardware — but with plain sign-bit quantization:

```
shortlist   100: recall@10 = 0.220
shortlist   300: recall@10 = 0.407
shortlist  1000: recall@10 = 0.633
```

0.63 is not shippable. The cause is structural: clustered vectors share most of their sign
bits, so Hamming distance can't discriminate *inside* a cluster — exactly where the top-k
lives. The fix is a random rotation before binarizing (the RaBitQ approach, SIGMOD 2024).
That's future work and explicitly research-risk.

**4-bit quantization is dead.** Its bound is valid but prunes essentially nothing, so 8x
compression buys a full scan followed by a full rescore. Measured, then deleted.

---

## Features

- **Exact k-NN**, cosine similarity, dimensions configured at startup.
- **Provably lossless pruning** — results bit-identical to brute force, enforced in CI.
- **Struct-of-arrays layout**, unit-normalized at insert so cosine *is* dot product.
- **Free-list deletes** — no compaction, no background goroutine, no RCU.
- **Bounded min-heap top-k**, pooled, one per worker, merged at the end.
- **Parallel scan** across `GOMAXPROCS`.
- **Crash-consistent snapshots** — `tmp → fsync → rename → fsync(parent dir)` with a CRC32.
- **Write-ahead log** — a write is acknowledged only once its record is on disk, with
  group commit so concurrent writers share one `fsync`. On by default wherever
  snapshots are.
- **Arbitrary byte-slice payloads** returned on search.

**Memory:** `capacity × dims × 5 bytes` + payloads — ≈368 MiB at 100k × 768, allocated
eagerly at boot so the process fails immediately rather than under load.

**Requirements:** Go 1.21+, amd64 or arm64 (and anything else Go targets, on the pure-Go
fallback). AVX2 + FMA3 for the speedup (Haswell 2013+ / Zen 2017+); without it MinDB runs
correctly on the fallback and logs the active kernel at startup. arm64 is on the fallback
today — see the note under the benchmarks.

---

## Wire protocol

gRPC over HTTP/2 with FlatBuffers, defined in [`fbs/mindb.fbs`](../fbs/mindb.fbs):

```flatbuffers
table Vector          { id: string; values: [float32]; payload: [ubyte]; }
table SearchRequest   { query_vector: [float32]; top_k: int32; }
table SearchResult    { id: string; score: float32; payload: [ubyte]; }

rpc_service VectorService {
  Insert(InsertRequest):     InsertResponse;
  Search(SearchRequest):     SearchResponse;
  Delete(DeleteRequest):     DeleteResponse;
  Snapshot(SnapshotRequest): SnapshotResponse;
  Get(GetRequest):           GetResponse;
  Stats(StatsRequest):       StatsResponse;
}
```


**`Get` is a point lookup, not a search.** It returns the stored record for each
id in request order; ids that are absent — never inserted, or deleted — are
omitted rather than reported as an error, so one unknown id cannot fail a batch.
The response is therefore **not positionally aligned with the request**: match on
id, not on index.

What comes back is the unit-normalized vector, because that is what `Insert`
stored. Normalization discards the original magnitude and MinDB does not keep a
copy, so `Get` returns `v/|v|`, not `v`. The values are exact — they come from
the float32 slab, not from the int8 codes, which only ever prune search
candidates.

**`Stats` reports engine state and which kernel is live** — vector count,
capacity, dims, reserved bytes, live payload bytes, plus `kernel_name`,
`fast_int8` and `goarch`. The last three exist because *which search path a
deployment is actually running* is otherwise visible only in the startup log of
a process nobody has a terminal for, and on ARM it is the difference between the
cascade and a brute-force scan.

It also reports `wal_enabled` and `wal_healthy`. The second is the readiness
signal: a node whose log has faulted still answers searches correctly, so nothing
about liveness catches it, and this is what says stop sending it writes. The
server drives a `grpc.health.v1.Health` service off the same state, on its own
port.

**On "zero-copy":** half of it is real, and it's worth being precise about which half.
*Reading* a request genuinely is zero-copy — the buffer arrives little-endian and 4-byte
aligned, so the query vector is reinterpreted in place rather than read through
bounds-checked per-element accessors. *Building* a response allocates. FlatBuffers doesn't
change that.

---

## Roadmap

| stage | contents | state |
|---|---|---|
| 1 | correct exact engine — math, core, gRPC, server | done |
| 2 | bound-and-refine cascade, differential test | done |
| 3 | Avo-generated AVX2 int8 kernel | done |
| 4 | rotated 1-bit tier (RaBitQ-style) | deferred, research-risk |
| — | NEON int8 kernel for arm64 | planned |
| — | write-ahead log | done |
| — | segmented store: write buffer, immutable segments, tombstones | planned |

Stage 2 is expected to be *slower* than stage 1. Its job is to prove correctness before
any assembly exists to blame for a wrong answer.

**On the segmented store:** MinDB today is a flat slab — parallel arrays sized at
boot, an id map, a free list that recycles deleted slots, whole-file snapshots and
a log that covers the gap between them. That is a real design, not a stopgap: it
makes a scan sequential and deletes compaction entirely. What it does not give is
a write path that does not contend with readers on a single lock, or an index that
grows past the capacity reserved at boot. Those are what a segmented store is for,
and it is its own track with its own spec rather than something to bolt onto the
slab.

---

## Project layout

```
mindb/
├── cmd/mindb-server/       # boots gRPC server, loads snapshot
├── pkg/
│   ├── api/                # FlatBuffers gRPC service implementation
│   ├── core/               # engine (RWMutex, free list, cascade), snapshots
│   ├── math/               # dot/normalize kernels + generated AVX2 assembly
│   └── mindb/              # flatc-generated — do not hand-edit
├── fbs/mindb.fbs           # the wire contract (frozen)
└── docs/ARCHITECTURE.md    # design record, measurements, rejected approaches
```

---

## Honest limits

MinDB is **not** a replacement for HNSW at scale. It is exact, so its cost grows linearly
with `N × dims`; a graph index is sublinear and will beat it well before a million
vectors. Reach for MinDB when you want exact answers on a bounded corpus with no recall
tuning, no index build time, and no approximation to explain to anyone. Reach for HNSW
when `N` is large and 0.97 recall is fine.

Writes block behind in-flight scans. There is no replication, no sharding, and no
multi-tenancy. Capacity is fixed at boot.
