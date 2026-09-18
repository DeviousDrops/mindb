# MinDB — feature reference

This is the detailed, feature-by-feature reference for MinDB. For the
project pitch and measured numbers, see [`docs/README.md`](docs/README.md).
For *why* things are built this way, see [`DECISIONS.md`](DECISIONS.md) and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). If you're new to Go and want
an ordered path through the source, see [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md).

MinDB is an embedded, in-memory, **exact** k-NN vector search engine
exposed over gRPC. "Exact" is the load-bearing word: every search result is
bit-identical to a brute-force scan, proven by construction rather than
tuned for.

## Table of contents

- [Engine (`pkg/core`)](#engine-pkgcore)
  - [Creating an engine](#creating-an-engine)
  - [Insert](#insert)
  - [Delete](#delete)
  - [Search](#search)
  - [Get](#get)
  - [Stats](#stats)
  - [Concurrency model](#concurrency-model)
  - [Slot allocation and the free list](#slot-allocation-and-the-free-list)
- [Search algorithms](#search-algorithms)
  - [Plain scan](#plain-scan)
  - [The bound-and-refine cascade](#the-bound-and-refine-cascade)
- [Quantization (`pkg/math`)](#quantization-pkgmath)
  - [Dot, Norm, Normalize](#dot-norm-normalize)
  - [Quantize](#quantize)
  - [DotInt8 and the AVX2 kernel](#dotint8-and-the-avx2-kernel)
- [Snapshots (durability)](#snapshots-durability)
  - [File format](#file-format)
  - [Save](#save)
  - [Load](#load)
- [gRPC API (`pkg/api`)](#grpc-api-pkgapi)
  - [Wire schema](#wire-schema)
  - [Insert RPC](#insert-rpc)
  - [Search RPC](#search-rpc)
  - [Delete RPC](#delete-rpc)
  - [Snapshot RPC](#snapshot-rpc)
  - [Get RPC](#get-rpc)
  - [Stats RPC](#stats-rpc)
  - [The zero-copy read path](#the-zero-copy-read-path)
- [Server (`cmd/mindb-server`)](#server-cmdmindb-server)
  - [Flags](#flags)
  - [Boot sequence](#boot-sequence)
  - [Shutdown sequence](#shutdown-sequence)
- [Testing strategy](#testing-strategy)

---

## Engine (`pkg/core`)

`Engine` (`pkg/core/engine.go`) is the whole database: an in-memory,
struct-of-arrays store of unit-normalized vectors, keyed by a string ID,
with an optional byte-slice payload attached to each vector.

```go
type Engine struct {
    dims, capacity int
    mu             sync.RWMutex
    vectors        []float32 // capacity*dims, unit-normalized
    codes          []int8    // capacity*dims, int8 quantization of vectors
    scales         []float32 // capacity, per-vector quantization scale
    residuals      []float32 // capacity, ρ = ‖v − v̂‖, the pruning bound
    externalID     []string
    payloads       [][]byte
    live           []bool
    idMap          map[string]uint32
    free           []uint32
    highWater      uint32
    count          int
    useCascade     bool
    heaps, scratch sync.Pool
}
```

Every field beyond `dims`/`capacity`/`mu` is sized to `capacity` and
allocated once, eagerly, at construction — see [Creating an
engine](#creating-an-engine).

### Creating an engine

```go
func New(dims, capacity int) (*Engine, error)
```

- `dims` — the vector dimension every `Insert`/`Search` call must match.
- `capacity` — the maximum number of *live* vectors the engine can hold.
  Once `capacity` is exhausted, `Insert` returns `ErrCapacityExceeded` until
  something is deleted.

Allocation is **eager**: `New` immediately allocates every backing slice at
full `capacity`. At `dims=768, capacity=100_000` that's ≈368 MiB (see
`estimateRAM` in `cmd/mindb-server/main.go` for the exact formula:
`capacity × (dims×5 + 8)` bytes, covering the float32 vector, the int8 code,
and the per-vector scale/residual floats). The intent is that a
misconfigured deployment fails at boot with an out-of-memory error, not
partway through inserting live traffic.

`New` also decides whether the cascade search path is available, by calling
`math.HasFastInt8()` — see [The bound-and-refine
cascade](#the-bound-and-refine-cascade).

Errors: `dims <= 0` or `capacity <= 0` return a descriptive `error`
(`fmt.Errorf`, not one of the sentinel `Err*` vars — those are reserved for
per-operation failures below).

```go
func (e *Engine) SetCascade(on bool)  // force the search path, for benchmarking
func (e *Engine) Cascade() bool       // is the cascade path currently active?
func (e *Engine) Dims() int
func (e *Engine) Cap() int
func (e *Engine) Len() int            // live vector count
```

`SetCascade`/`Cascade` exist because both paths return *identical* results —
switching between them is a performance knob, never a correctness one. This
is what the differential tests in `cascade_test.go` exploit: run the same
query through both paths and assert the outputs match exactly.

### Insert

```go
func (e *Engine) Insert(id string, vec []float32, payload []byte) error
```

What happens, in order:

1. Rejects `id == ""` with `ErrEmptyID`.
2. Rejects `len(vec) != Dims()` with `ErrDimensionMismatch`.
3. **Copies** `vec` into a fresh buffer and normalizes the copy in place
   (`math.Normalize`). Rejects an all-zero vector with `ErrZeroVector` — a
   zero vector has no direction, and dividing by its zero norm would produce
   `NaN`s that then silently poison every future score involving that slot.
4. Copies `payload` (if non-empty) into a fresh buffer.
5. Takes the write lock and either reuses the existing slot for `id` (if
   `id` was already present — this is how you *update* a vector: insert
   again under the same ID) or allocates a new one via the free list.
6. Stores the normalized vector, recomputes its int8 quantization
   (`math.Quantize`), and stores the payload.

**Why the copies matter:** the gRPC handler path hands `Insert` a slice that
points directly into a FlatBuffers receive buffer gRPC will recycle the
instant the handler returns (see [zero-copy read
path](#the-zero-copy-read-path)). If `Insert` didn't copy, the stored vector
could be silently overwritten by an unrelated future RPC. Both `vec` and
`payload` are safe for the caller to mutate or discard immediately after
`Insert` returns — nothing is retained by reference.

Re-inserting under an existing ID **replaces** that vector (and its
payload) in place; it does not create a duplicate or require a prior
`Delete`.

**Normalization is lossy and the loss is not recoverable.** Only `v/|v|` is
stored; `|v|` is computed, used to divide, and thrown away. Nothing in the
engine keeps a copy, so [`Get`](#get) returns the unit vector, not the one
you sent. For cosine similarity — the only thing MinDB scores — the two are
interchangeable, which is why storing the norm (4 bytes per vector plus a
snapshot format version) has not been worth it. If a caller needs the
magnitude, it keeps its own copy.

### Delete

```go
func (e *Engine) Delete(id string) bool
```

Returns whether `id` existed. On success, the slot is marked not-live, its
ID and payload references are cleared (so the payload's backing array can be
garbage-collected even if a caller elsewhere still holds a `Result` that
aliased it — see the aliasing note in [Search](#search)), and the slot index
is pushed onto the free list for reuse by a future `Insert`.

There is no compaction, no background goroutine, and no reference counting.
See [`DECISIONS.md`](DECISIONS.md#decision-replace-rcu-compaction-with-a-free-list)
for why.

### Search

```go
func (e *Engine) Search(query []float32, k int) ([]Result, error)

type Result struct {
    ID      string
    Score   float32
    Payload []byte
}
```

- `k <= 0` returns `(nil, nil)` — not an error, just "you asked for nothing."
- `k` larger than the live count is silently clamped down to the live count.
- `query` is copied and normalized internally (same zero-vector rejection as
  `Insert`); the caller's slice is never modified.
- Results are always exact and always sorted **best first**, with ties
  broken by ascending external ID (not slot — slots aren't stable across a
  snapshot reload, IDs are).
- `Result.Payload` is **not** a copy — it aliases the engine's internal
  payload slice. This is a deliberate, documented trade: copying every
  payload on every search would be wasted work for the overwhelmingly common
  case where the caller only reads the result and discards it. The
  concurrency model makes this safe rather than merely convenient: a
  concurrent `Delete` clears the engine's *own* reference to the payload but
  cannot invalidate a `Result` a caller is already holding — Go's garbage
  collector keeps the backing array alive as long as any reference exists.
  Worst case is a stale read of data that's since been logically deleted,
  never a use-after-free. If you need to retain a `Result.Payload` past the
  scope where you'd otherwise let the `Engine` continue mutating, copy it
  yourself.

Internally, `Search` dispatches to either [`scan`](#plain-scan) (plain
float32) or [`searchCascade`](#the-bound-and-refine-cascade) depending on
`Engine.useCascade`, under a single `RLock` held for the whole operation.

### Concurrency model

One `sync.RWMutex` (`e.mu`), full stop. Reads (`Search`, `Len`, `Cascade`)
take `RLock`; writes (`Insert`, `Delete`, `SetCascade`) take the exclusive
`Lock`. Multiple `Search` calls run genuinely concurrently — `RLock` doesn't
serialize them — but a `Search` in flight blocks a concurrent `Insert`/
`Delete`, and vice versa.

Full rationale, including the three concrete bugs in the lock-free design
this replaced, is in
[`DECISIONS.md`](DECISIONS.md#concurrency).

### Slot allocation and the free list

```go
func (e *Engine) allocSlot() (uint32, error)
```

Not exported — an internal helper, called with the write lock already held.
Pops from `e.free` if non-empty; otherwise takes the next unused index at
`e.highWater` and advances it. Returns `ErrCapacityExceeded` once
`highWater` reaches `capacity` and the free list is empty.

`highWater` is also what bounds every scan: `scan`/`scanCascade` iterate
`[0, highWater)` and skip non-live slots, rather than iterating a possibly
much larger `capacity`. This means a workload that inserts N vectors and
deletes half of them still scans through N slots (skipping the dead ones),
not `N/2` — a known, documented tradeoff of the free-list design; it never
shrinks `highWater` back down.

---

### Get

```go
func (e *Engine) Get(ids []string) []Record

type Record struct {
    ID      string
    Vector  []float32
    Payload []byte
}
```

Point lookup by external ID, batched. Returns one `Record` per id that exists,
**in request order**. Ids that were never inserted, or that have been deleted,
are simply absent from the result — there is no error and no placeholder, so
`len(result) < len(ids)` is normal and **the result is not positionally aligned
with the request**. A caller that needs to know which ids missed compares the
returned ids against the ones it asked for.

The whole batch is served under a single `RLock`, so it observes one consistent
instant. Locking per id would be no cheaper and would let a batch see a state
that never existed.

**What you get back is not what you inserted.** `Insert` normalizes and discards
the original norm (see [Insert](#insert)), so `Vector` is `v/|v|`. It is exact,
though: it is read from the float32 slab, never reconstructed from the int8
codes, which exist only to prune search candidates and are never a source of
truth.

**Both slices are copies the caller owns.** Returning a sub-slice of the slab
would let a later `Insert` — possibly under a different id that inherited the
slot from the free list — rewrite a response the caller is still holding, after
the read lock has dropped. One allocation per hit is the price of a result that
stays valid.

### Stats

```go
func (e *Engine) Stats() Stats

type Stats struct {
    Count    int // live vectors
    Capacity int // slots allocated at boot
    Dims     int

    MemoryBytes  int64
    PayloadBytes int64

    KernelName string
    FastInt8   bool
    GoArch     string
}
```

A point-in-time report, taken under the read lock.

`MemoryBytes` is the slab footprint **reserved**, not used: allocation is eager,
so it is the same number the moment after `New` as it is at capacity. It is
`capacity * (dims*5 + 8)` — `dims*4` for the float32 copy, `dims*1` for the int8
code, and 4 bytes each for the scale and the residual norm. Ids, payloads and
the id map sit on top of it.

`PayloadBytes` is the live total, payloads being the only variable-size
allocation the engine owns. It is a running counter updated by every path that
stores or drops a payload, not a sum computed on demand: walking every live slot
would make `Stats` O(capacity), which is backwards for a call a monitor polls.

`KernelName`, `FastInt8` and `GoArch` answer *which search path is this
deployment actually running*. On amd64 with AVX2 that is `avx2` and the cascade;
on arm64 it is `pure-go` and a brute-force scan (see
[DotInt8 and the AVX2 kernel](#dotint8-and-the-avx2-kernel)). The same three
values are logged at startup.

## Search algorithms

Both algorithms live under the same `Search` entry point and are switched by
`Engine.useCascade`. They are required to return **identical** output —
enforced by `TestCascadeIsExact` in `pkg/core/cascade_test.go`, described in
its own comment as "the centerpiece of the whole project."

### Plain scan

`pkg/core/engine.go: scan`, `scanRange`

The straightforward approach: compute `math.Dot(query, vector)` for every
live vector, keep the top `k` in a pooled min-heap. Below
`parallelScanThreshold` (8192 live vectors) this runs on the calling
goroutine; above it, the range `[0, highWater)` is split into
`GOMAXPROCS`-many contiguous chunks, each scored on its own goroutine into
its own pooled heap, and the heaps are merged once all workers finish.

This is the fallback path — always correct, used unconditionally when
`useCascade` is false (either because the hardware lacks AVX2, or because a
caller explicitly disabled it via `SetCascade(false)`).

### The bound-and-refine cascade

`pkg/core/cascade.go: searchCascade`, `scanBounds`, `boundRange`

The performance path, three passes:

**Pass 1 — bound.** Score every live vector against its *int8 code* (not the
full float32 vector) using `math.DotInt8`, scaled by that vector's per-slot
`scale`. This produces an *approximate* score `approx[slot]`. Simultaneously
track the `k` largest values of `approx[slot] − residual[slot]` — call the
smallest of those `τ` (tau).

**Why `τ` is a safe threshold, not a heuristic:** every vector's *true*
score lies within `[approx − ρ, approx + ρ]` (Cauchy-Schwarz, given
unit-normalized vectors and a float32 query — see
[Quantize](#quantize)). At least `k` vectors have a lower bound `≥ τ` by
construction (`τ` is literally the `k`-th largest lower bound), so at least
`k` vectors have a *true* score `≥ τ`. Therefore the true top-k, whichever
vectors they turn out to be, all have true score `≥ τ`, and since every
vector's upper bound is `≥` its true score, every member of the true top-k
also has upper bound `≥ τ`. A vector with upper bound `< τ` *cannot* be in
the true top-k. This is the entire proof; nothing here is a probability or
an expected recall.

**Pass 2 — filter.** One cheap comparison per live slot:
`approx[slot] + residual[slot] >= τ`. Survivors go into a slice. This is
just two float32 reads and a compare per slot — no vector data touched, no
allocation beyond the survivor slice itself.

**Guard fallback:** if survivors exceed `guardFraction` (0.25) of the live
count, abandon the cascade and run a plain [`scan`](#plain-scan) instead —
scattered exact rescoring of a large survivor set can cost more than one
straight parallel scan would have. See
[`DECISIONS.md`](DECISIONS.md#decision-guard-fallback-to-a-plain-float32-scan-when-survivors-exceed-25-of-the-corpus)
for how the constant was chosen. In practice, on realistic (clustered) data,
survivor rates are ~0.4% — the guard almost never fires.

**Pass 3 — refine.** Compute `math.Dot` (the exact float32 kernel) for every
surviving slot only, and keep the top `k` in a heap. This is the only pass
that touches full-precision vector data, and it only touches the (typically
tiny) survivor set.

`searchCascade` returns both the result heap and the survivor count, which
tests use to assert pruning is actually effective
(`TestCascadePruningIsEffective`) and not just correct.

---

## Quantization (`pkg/math`)

### Dot, Norm, Normalize

`pkg/math/distance.go`

```go
func Dot(a, b []float32) float32       // panics if len(a) != len(b)
func Norm(v []float32) float32          // Euclidean length, float64 accumulation
func Normalize(v []float32) float32     // scales v to unit length in place; returns original norm, or 0 for a zero vector
```

`Dot` is unrolled by 8 with independent accumulators — not to reduce
instruction count, but because a single accumulator is a serial dependency
chain (each add waits on the previous one's latency); 8 independent chains
let the CPU's out-of-order execution engine overlap them, which is where
the ~1.7x speedup over a naive scalar loop comes from. The accumulator
summation order (`((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7))`) is part of the
function's contract, not an implementation detail — floating-point addition
isn't associative, so changing the order changes results in the last bits,
and the cascade's differential tests compare scores for *exact* equality.

`Norm` accumulates in `float64` even though the vectors are `float32`,
because `Normalize`'s output feeds directly into the cascade's correctness
bound — precision lost here isn't just a slightly-off score, it's a
weakened guarantee.

### Quantize

`pkg/math/quantize.go`

```go
func Quantize(v []float32, code []int8) (scale, residual float32)
```

Encodes a unit-length vector `v` into `code` (which must be pre-allocated to
`len(v)`) using symmetric int8 quantization: find the largest-magnitude
component, map it (and everything else, scaled proportionally) onto
`[-127, 127]` — 127, not 128, so the positive and negative ranges are
symmetric and rounding introduces no directional bias.

Returns:
- `scale` — the float32 that satisfies `v̂ᵢ = codeᵢ × scale` for the
  reconstructed vector `v̂`.
- `residual` (`ρ`) — the *measured* Euclidean distance `‖v − v̂‖` between the
  original vector and its quantized reconstruction. This is measured
  directly with a second pass over `v`, not derived algebraically from
  `scale` — see [`DECISIONS.md`](DECISIONS.md#decision-residual-norm-is-measured-directly-not-derived-from-the-scale)
  for why the algebraic shortcut was tried and rejected.

An all-zero input vector yields `scale = 0`, `residual = 0`, and an
all-zero code — a degenerate but well-defined case, distinct from `Normalize`
rejecting zero vectors at the `Engine` layer (by the time `Quantize` is
called from `Insert`, the zero-vector case has already been rejected; the
zero-input behavior here exists mainly so `Quantize` is well-defined as a
standalone function and doesn't panic on an edge case its caller has already
ruled out upstream).

Panics if `len(code) != len(v)`.

### DotInt8 and the AVX2 kernel

`pkg/math/quantize.go`, `pkg/math/kernel.go`, `pkg/math/kernel_amd64.go`,
`pkg/math/kernel_generic.go`, `pkg/math/avo/asm.go`,
`pkg/math/dotint8_avx2_amd64.s`

```go
func DotInt8(q []float32, code []int8) float32  // panics if len(q) != len(code)
func HasFastInt8() bool                          // is a SIMD kernel active?
func KernelName() string                         // "avx2" or "pure-go", for logging
```

`DotInt8` computes the dot product of a float32 query against an int8 code
— the "asymmetric" half of the cascade (only one side is quantized). The
public function validates lengths and then dispatches to `dotInt8Impl`, a
package-level function variable resolved once, at `init()`:

- **On amd64** (`kernel_amd64.go`), `init()` checks
  `cpu.X86.HasAVX2 && cpu.X86.HasFMA && cpu.X86.HasAVX` via
  `golang.org/x/sys/cpu`. If all three are present, `dotInt8Impl` becomes
  `dotInt8AVX2` (the generated kernel) and `KernelName()` reports `"avx2"`.
  Otherwise it falls back to `dotInt8Generic` and reports `"pure-go"`.
- **On every other architecture** (`kernel_generic.go`), there is no SIMD
  path at all; `dotInt8Impl` is always `dotInt8Generic`.

`dotInt8Generic` is an 8-way-unrolled pure-Go loop, structurally identical
to `Dot` but multiplying a `float32` against a `float32(int8)` conversion
per lane. It is also the **correctness oracle**: every kernel, generated or
not, is validated against it (`TestDotInt8MatchesReference`,
`TestDotInt8AVX2MatchesGeneric`).

**The generated kernel** (`dotInt8AVX2`) is produced from
`pkg/math/avo/asm.go`, a `//go:build ignore` Go program using the
[Avo](https://github.com/mmcloughlin/avo) DSL. Regenerate it with:

```sh
cd pkg/math/avo
go run asm.go -out ../dotint8_avx2_amd64.s -stubs ../dotint8_avx2_amd64_stub.go -pkg math
```

Structurally: an unrolled-by-4 block loop, each unrolled lane processing 8
lanes per iteration (32 elements per block) —

1. `VPMOVSXBD` sign-extends 8 packed `int8` code bytes into 8 `int32` lanes
   of a 256-bit YMM register.
2. `VCVTDQ2PS` converts those 8 `int32` lanes to 8 `float32` lanes, in
   place.
3. `VFMADD231PS` multiplies those 8 floats against 8 float32 query lanes and
   accumulates into one of 4 independent YMM accumulators (same
   independent-chain reasoning as `Dot`'s unroll).

A scalar tail loop (`MOVBLSX` → `VCVTSI2SSL` → `VFMADD231SS`) handles
whatever's left when the remaining length isn't a multiple of 32, and a
final horizontal reduction (`VADDPS`/`VEXTRACTF128`/`VHADDPS`×2) collapses
the 4 accumulators plus the tail into the single `float32` return value.

Why AVX2+FMA3 rather than AVX-512, and why Avo rather than hand-written
assembly, are covered in
[`DECISIONS.md`](DECISIONS.md#the-avx2-kernel-stage-3).

**Measured** on the AVX2 kernel (`BenchmarkDotInt8`, this machine): ~6.5
GB/s at 128 dims, ~23.5 GB/s at 768 dims — comfortably above this machine's
memory bandwidth ceiling (~11.7 GB/s), meaning the int8 scan is now
bandwidth-bound rather than compute-bound. That's the entire point: pure-Go
int8 plateaus at ~0.83 G MAC/s, which is *slower* than just scanning
float32 directly, so without this kernel the cascade would never be worth
taking (`Engine.useCascade` gates on exactly this).

---

## Snapshots (durability)

`pkg/core/snapshot.go`

### File format

All integers little-endian:

```
magic     8 bytes   "MINDBSNP"
version   uint32    currently 1
dims      uint32
count     uint32
records   count × { idLen uint32, id, payloadLen uint32, payload, dims×float32 }
crc32     uint32    IEEE checksum over every byte above
```

Only the float32 vector is stored per record — int8 codes, scales, and
residuals are recomputed on load. See
[`DECISIONS.md`](DECISIONS.md#decision-int8-codes-are-not-persisted-in-the-snapshot).

### Save

```go
func (e *Engine) Save(path string) error
```

Takes a read lock for the duration of the write (a consistent point-in-time
snapshot; concurrent writers block until it finishes), streams every live
record through a buffered writer that also feeds a running CRC32, and
writes the checksum last. Sequence for atomicity:

```
write to path+".tmp*" → fsync the tmp file → close it → rename over path → fsync the containing directory
```

The final directory fsync is the step most hand-rolled atomic-write code
skips; without it, a crash immediately after the rename can leave the
directory entry unpointed even though the renamed file's bytes are safely
on disk. On Windows, which has no directory-fsync equivalent, that last
step is a documented no-op (`runtime.GOOS == "windows"`) rather than a
silent gap.

On any failure partway through, the temp file is removed rather than left
behind — `Save` never leaves a `.tmp*` file on disk on error, which is
asserted directly by `TestSnapshotRoundTrip`'s sibling tests in
`snapshot_test.go`.

### Load

```go
func Load(path string, capacity int) (*Engine, error)
```

Reads and validates the header (magic, version), builds a fresh `Engine`
sized to `capacity` with `dims` taken from the file (not from the caller —
a caller-supplied `dims` in `cmd/mindb-server` is only used when *no*
snapshot exists), then reads each record and calls the internal `store()`
(not `Insert()`) to avoid re-normalizing already-unit vectors — see
[`DECISIONS.md`](DECISIONS.md#decision-load-calls-the-internal-store-not-the-public-insert).
The checksum is verified last, over every byte read; a mismatch returns
`ErrBadChecksum` and the partially-built engine is discarded.

Returns `ErrSnapshotSize` if the file declares more vectors than `capacity`
allows, `ErrBadMagic`/`ErrBadVersion` for a file that isn't a recognizable
MinDB snapshot at all, and a wrapped error naming the specific record index
for any truncation mid-file.

A vector's slot index is **not** preserved across a reload — slots are
reassigned in file order — but its external ID is, which is the identifier
callers should treat as stable.

---

## gRPC API (`pkg/api`)

`Server` (`pkg/api/grpc_server.go`) implements the generated
`mindb.VectorServiceServer` interface over a `*core.Engine`.

```go
func New(engine *core.Engine, snapshotPath string) *Server
```

`snapshotPath` may be empty; the `Snapshot` RPC then reports persistence as
disabled rather than erroring, so a caller can safely call it unconditionally
without knowing the server's configuration.

### Wire schema

`fbs/mindb.fbs`:

```flatbuffers
table Vector          { id: string; values: [float32]; payload: [ubyte]; }
table InsertRequest   { vectors: [Vector]; }
table InsertResponse  { inserted_count: int32; }

table SearchRequest   { query_vector: [float32]; top_k: int32; }
table SearchResult    { id: string; score: float32; payload: [ubyte]; }
table SearchResponse  { results: [SearchResult]; }

table DeleteRequest   { ids: [string]; }
table DeleteResponse  { deleted_count: int32; }

table SnapshotRequest  {}
table SnapshotResponse { success: bool; message: string; }

table GetRequest      { ids: [string]; }
table GetResponse     { vectors: [Vector]; }

table StatsRequest    {}
table StatsResponse   { vector_count: uint32; capacity: uint32; dims: uint32;
                        memory_bytes: uint64; payload_bytes: uint64;
                        kernel_name: string; fast_int8: bool; goarch: string; }

rpc_service VectorService {
  Insert(InsertRequest):     InsertResponse;
  Search(SearchRequest):     SearchResponse;
  Delete(DeleteRequest):     DeleteResponse;
  Snapshot(SnapshotRequest): SnapshotResponse;
  Get(GetRequest):           GetResponse;
  Stats(StatsRequest):       StatsResponse;
}
```

The generated Go bindings live in `pkg/mindb/` — flatc-generated, never
hand-edit. `make gen` regenerates them with a `flatc` pinned to the same
version as the FlatBuffers runtime in `go.mod`; `make check-gen` fails when
the committed output and the schema have drifted apart. A version mismatch
between compiler and runtime does not fail the build — it shifts vtable
offsets and shows up as garbage field values at runtime — which is why the
pin exists.

### Insert RPC

Inserts every `Vector` in the request in order. **Not transactional**: if
vector `i` fails validation, vectors `0..i-1` are already committed to the
engine and stay committed; the RPC returns an error identifying which
vector failed, its underlying cause, and how many vectors before it were
already stored (`insertError`). Error causes map to gRPC status codes:

| Engine error | gRPC code |
|---|---|
| `ErrDimensionMismatch`, `ErrZeroVector`, `ErrEmptyID` | `InvalidArgument` |
| `ErrCapacityExceeded` | `ResourceExhausted` |
| anything else | `Internal` |

### Search RPC

Reads `query_vector` via the zero-copy path (below), rejects an empty query
vector with `InvalidArgument`, and otherwise delegates straight to
`Engine.Search`. Response results are appended in rank order; because
FlatBuffers vectors are built back-to-front, the handler prepends result
offsets in reverse to land them in the right order on the wire.

Response builders are **not** pooled — gRPC marshals the builder's buffer
after the handler returns, so there's no safe point at which to recycle it.
This is the documented "building a response still allocates" half of the
zero-copy story.

### Delete RPC

Deletes every ID in the request and returns how many actually existed
(`deleted_count`) — deleting a nonexistent ID is not an error, it's just not
counted.

### Snapshot RPC

Calls `Engine.Save` at the server's configured `snapshotPath`. Returns
`success: false` with an explanatory message rather than a gRPC error both
when persistence is disabled and when the save itself fails — a design
choice that keeps failure information in the response body rather than
requiring callers to parse gRPC status details for something that's really
just "did the file get written."

### Get RPC

Delegates to `Engine.Get` and encodes each record as a `Vector` — the same
table `Insert` accepts, so a `Get` result is wire-identical to an `Insert`
input and can be fed straight back in.

Ids that are absent are omitted rather than erroring, so `vectors` may be
shorter than the ids in the request and is not positionally aligned with
them; match on `id`. Found records come off the wire in request order, which
— as in [Search](#search-rpc) — means the handler prepends offsets in
reverse, FlatBuffers vectors being built back-to-front. Strings and nested
vectors are created before the table that references them, because
FlatBuffers forbids opening one inside the other.

A vector with no payload gets no `payload` field at all, rather than an empty
one.

### Stats RPC

Delegates to `Engine.Stats` and copies the fields onto the wire. Takes no
arguments and cannot fail.

`kernel_name`, `fast_int8` and `goarch` are the reason this RPC exists in a
deployment rather than just in a log line: they let the thing operating MinDB
report which search path is live, which on ARM is the difference between the
cascade and a brute-force scan.

There is no `segment_count`. MinDB is a flat slab, and a field that always
reported `1` would describe a system that does not exist; FlatBuffers can
append one when segments do.

### The zero-copy read path

`float32Vector(tab flatbuffers.Table, slot flatbuffers.VOffsetT) []float32`

The generated FlatBuffers accessor reads one `float32` per call through
bounds-checked offset arithmetic — correct, but a copy. Because a
FlatBuffers wire buffer is already little-endian, 4-byte-aligned `float32`
data, `float32Vector` instead validates the vector's byte range once and
then reinterprets that region directly as a `[]float32` via `unsafe.Slice` —
no per-element copy at all. A package-level `nativeLittleEndian` check
(evaluated once, at init) guards a byte-swapping fallback path for the
theoretical case where the host isn't little-endian, rather than silently
assuming it always is.

The returned slice **aliases the request's receive buffer**, which gRPC
recycles the moment the handler returns — this is why `core.Engine.Insert`
and `core.Engine.Search` both copy their input immediately rather than
retaining what's handed to them. This function is a genuine zero-copy
win only because its caller doesn't rely on it staying valid.

Vtable slot constants (`vectorValuesSlot = 6`, `searchQueryVectorSlot = 4`)
are hardcoded to match `fbs/mindb.fbs`'s field order because this path
bypasses the generated accessors that would otherwise track that for you —
if the schema's field order ever changes, these constants must change with
it.

---

## Server (`cmd/mindb-server`)

`cmd/mindb-server/main.go` — the deployable binary.

### Flags

| flag | default | meaning |
|---|---|---|
| `-addr` | `:50051` | gRPC listen address |
| `-dims` | `768` | vector dimension; ignored if a snapshot is loaded (the snapshot's own `dims` wins) |
| `-capacity` | `100000` | max vectors; memory for this is reserved eagerly at boot |
| `-snapshot` | `""` | snapshot file path; empty disables persistence entirely |
| `-snapshot-interval` | `0` | periodic auto-snapshot interval; `0` disables. Requires `-snapshot` to be set — the server refuses to start otherwise |

### Boot sequence

1. `open(dims, capacity, snapPath)`:
   - No `-snapshot` path given → fresh empty `core.New(dims, capacity)`.
   - Snapshot path given but the file doesn't exist → also a fresh empty
     engine (first run), logged as such.
   - Snapshot path given and loads successfully → `core.Load`, using the
     snapshot's own `dims` (a mismatched `-dims` flag is logged as a
     warning and overridden, not treated as fatal).
   - Snapshot path given but the file is **corrupt** (bad magic, bad
     checksum, truncated) → the server refuses to start at all, rather than
     silently booting empty. Booting empty here would look identical to
     genuine data loss from the outside; refusing to start is the honest
     failure mode.
2. Logs `dims`, `capacity`, loaded count, and estimated RAM.
3. Registers the gRPC server **with `grpc.ForceServerCodec(flatbuffers.FlatbuffersCodec{})`** — mandatory, since handlers return `*flatbuffers.Builder` rather than a type gRPC's default codec understands; omitting this fails at first request, not at startup.
4. Starts a background periodic-snapshot goroutine if `-snapshot-interval`
   is set.
5. Serves on `-addr` in a goroutine, then blocks on either a serve error or
   an OS interrupt/`SIGTERM` signal.

### Shutdown sequence

On `SIGINT`/`SIGTERM`: stop the periodic-snapshot goroutine, call
`srv.GracefulStop()` (finish in-flight RPCs, refuse new ones), *then* take
one final `Save` if a snapshot path is configured — deliberately ordered
after `GracefulStop` so an in-flight write RPC can't race the final
snapshot and get missed.

---

## Testing strategy

Every package's tests live alongside its source (`*_test.go`, standard Go
convention). The tests worth knowing about by name, because they encode the
project's actual correctness contract rather than incidental coverage:

- **`TestCascadeIsExact`** (`pkg/core/cascade_test.go`) — the cascade and
  the plain scan must return identical results across many random queries.
  This is the test that makes "provably lossless pruning" a claim backed by
  CI rather than a hope.
- **`TestBoundHoldsForRandomQueries`** (`pkg/math/quantize_test.go`) — the
  true score must always land inside `[approx−ρ, approx+ρ]`. This is the
  property the entire cascade's correctness rests on; if this ever fails,
  the cascade can silently drop correct answers.
- **`TestDotInt8AVX2MatchesGeneric`** (`pkg/math/kernel_avx2_test.go`) — the
  generated AVX2 kernel must agree with the pure-Go reference across
  dimension sizes that straddle its internal 32-element block/tail
  boundary, since that's exactly where a hand-written-assembly-generator
  bug would show up.
- **`TestConcurrentHammer`** (`pkg/core/engine_test.go`) — concurrent
  `Insert`/`Delete`/`Search` under load, aimed at the exact class of bug
  the RWMutex redesign exists to rule out.
- **`TestSnapshotRoundTrip`** and siblings (`pkg/core/snapshot_test.go`) —
  save, reload, and assert identical search results; separately assert that
  a corrupted checksum is rejected and that `Save` never leaves a temp file
  behind on failure.
- **`TestZeroCopyReadMatchesGeneratedAccessor`** (`pkg/api/grpc_server_test.go`)
  — the `unsafe.Slice`-based fast path must return exactly what the slow,
  bounds-checked generated accessor would.
