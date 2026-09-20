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
- [Write-ahead log](#write-ahead-log)
  - [What "write-ahead" means here](#what-write-ahead-means-here)
  - [Record format](#record-format)
  - [The write path, and group commit](#the-write-path-and-group-commit)
  - [Recovery](#recovery)
  - [Rotation, and the run id](#rotation-and-the-run-id)
  - [Health](#health)
  - [Opening an engine](#opening-an-engine)
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
  - [Version](#version)
  - [Container image](#container-image)
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
func (e *Engine) Delete(id string) (bool, error)
```

The bool is whether `id` existed — deleting an ID that isn't there is not an
error, it just returns `false`. The error is a durability failure: with the
[write-ahead log](#write-ahead-log) on, `Delete` does not return until the
record is on disk, and a log that cannot be written fails the call rather
than acknowledging a deletion a restart would undo. Deletes that change
nothing are not logged at all, so a caller retrying one in a loop cannot
grow the log.

On success, the slot is marked not-live, its
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

One `sync.RWMutex` (`e.mu`) guards the slab. Reads (`Search`, `Len`,
`Cascade`) take `RLock`; writes (`Insert`, `Delete`, `SetCascade`) take the
exclusive `Lock`. Multiple `Search` calls run genuinely concurrently —
`RLock` doesn't serialize them — but a `Search` in flight blocks a
concurrent `Insert`/`Delete`, and vice versa.

Full rationale, including the three concrete bugs in the lock-free design
this replaced, is in
[`DECISIONS.md`](DECISIONS.md#concurrency).

A second, plain `sync.Mutex` (`e.writeMu`) arrived with the [write-ahead
log](#write-ahead-log), and it does one job: hold apply-then-append together
so that the order records reach the log is the order they reached the slab.
Without it, two writers updating the same ID can land in the slab in one
order and in the log in the other, and a replay then rebuilds a database
that disagrees with the one that crashed. It is held across `store` and the
buffered append only — never across the `fsync`, which is where all the time
goes, and releasing it there is what lets [group
commit](#the-write-path-and-group-commit) work at all. Nothing takes it when
no log is attached.

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

    WALEnabled bool
    WALHealthy bool
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

`WALEnabled` says whether writes are being logged at all. `WALHealthy` is
false once a log write or `fsync` has failed and never goes back on — it is
what a readiness probe watches, and it is read *before* the lock is taken
rather than inside it, because the log has its own. Both are `true`/`true`
on a healthy logging engine and `false`/`true` when there is no log: nothing
has failed, there is just nothing to fail. See [Health](#health).

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

## Write-ahead log

`pkg/core/wal.go`, `wal_format.go`, `wal_replay.go`, `meta.go`, `open.go`

A snapshot on its own loses every write taken since it was written. The log
closes that window: an operation is acknowledged only once its record is on
disk, so **every acknowledged write survives a crash**, and what a snapshot
does not cover is replayed at the next boot.

### What "write-ahead" means here

Not what the name says. The engine applies an operation to the slab first
and appends the record second — the ordering is Redis's AOF, not ARIES. The
name is kept because that is what everyone calls the file.

The guarantee above is unaffected. What differs is the other side of it: an
*unacknowledged* write becomes visible to readers before it is durable, so a
`Search` running concurrently with an in-flight `Insert` can return a vector
a crash then erases. No client was ever told that write succeeded — from any
client's point of view it was in flight, and an in-flight write may or may
not be observed and may or may not survive. A client that needs to know
waits for the acknowledgement.

True log-before-apply would mean reserving the slot before the append, since
`store`'s only failure (`ErrCapacityExceeded`) is discovered while applying
and by then the record would already be durable. That is a different write
path, not a reordering of this one — see
[`DECISIONS.md`](DECISIONS.md#decision-apply-then-log-then-fsync-then-acknowledge--a-redo-log-not-write-ahead-ordering).

### Record format

The log is a sequence of numbered segments, `<base>.000001` and up, listed
and replayed in lexical order — the zero padding makes that numeric order.
All integers little-endian:

```
segment header
  magic     8 bytes   "MINDBWAL"
  version   uint32    currently 1
  dims      uint32    must match the engine; a mismatch refuses to start
  run id    16 bytes  ties this segment to the snapshot it follows

record
  len       uint32    length of body
  crc       uint32    crc32-IEEE over the four len bytes and the body
  body      len bytes

body, insert (op = 1)
  op        uint8
  idLen     uint32, id
  payLen    uint32, payload
  values    dims × float32, unit-normalized exactly as stored

body, delete (op = 2)
  op        uint8
  idLen     uint32, id
```

Two details that are not arbitrary:

- **The checksum covers the length field, not just the body.** The length is
  read first and decides how many bytes are read next, so a checksum over
  the body alone validates whatever the corrupt length said to read — the
  failure then gets reported at the wrong offset, after an implausible
  allocation on the way. `walMaxRecord` (256 MiB) bounds that allocation for
  the case where the damage is caught but the length was read first anyway.
- **Vectors are logged already normalized**, matching the snapshot, so
  replay can go through the internal `store()` rather than `Insert()`.
  Re-running `Normalize` on a unit vector is not the identity in float32, so
  logging the caller's vector instead would make a recovered engine return
  subtly different scores from the process that crashed — the same reason
  [`Load`](#load) does it this way.

### The write path, and group commit

`Insert` and `Delete` both do: encode the record outside any lock, take
`writeMu`, apply, buffer the record, release `writeMu`, then wait for the
record to become durable. A `Delete` that changes nothing skips the log
entirely.

The wait is where group commit lives. Every buffered record gets a sequence
number; a writer is finished when the durable sequence number reaches its
own, whoever got it there. The writer that arrives while no sync is in
flight becomes the leader: it flushes everything buffered so far, records
how far that reaches, and runs the `fsync` **with the lock released**, so
writers arriving meanwhile keep appending and land in the next batch.

Nothing is timer-driven. A lone writer syncs immediately and pays no added
latency; under load the batch is however many writers showed up during the
last `fsync`, so the system self-tunes to the disk it is on.
`BenchmarkWALInsert` measures all three arms — no log, one `fsync` per
write, and group commit — and the numbers are in
[`DECISIONS.md`](DECISIONS.md#decision-group-commit-with-a-leader-and-no-timer).

### Recovery

Replay walks the retained segments in order and applies each record through
the engine's internal write path, so the log being replayed is not written
back to itself. Replay is idempotent: insert and delete are blind writes
with no read-modify-write, so re-applying a range a snapshot already covers
converges on the same state. That is what makes the overlap left by a crash
mid-`Save` harmless.

A partial record at the end of the file is how a log is *expected* to end —
a process killed mid-append — so replay stops there and the open succeeds.
What matters is that the decision sticks: the damaged segment is truncated
to the last good record and every segment after it is deleted, so the next
restart cannot replay what this one discarded. Two recoveries over the same
files can never disagree about what the database contains.

A segment shorter than its header is the same case one step earlier: a crash
between creating the file and syncing its header. No record can exist in it,
so it is removed rather than treated as an unreadable log.

`WALRecovery` is what replay reports back, for the startup log:

```go
type WALRecovery struct {
    Segments   int
    Records    int
    Truncated  bool  // the log ended in damage; that record and everything after it is gone
    LegacyMeta bool  // a snapshot with no .meta beside it, so the log could not be checked
}
```

### Rotation, and the run id

The failure this prevents is an operator restoring last week's snapshot next
to this morning's log and having the log replayed over a history it never
belonged to. Nothing about those two files looks wrong on its own.

So every completed `Save` mints a 16-byte run id, stamps it into the segment
it rotates to, and records it in a companion `<snapshot>.meta` file. Recovery
accepts a log only when **some retained segment carries the id the meta
names** — exactly the condition "the log reaches back to at least this
snapshot". A mismatch refuses to start, with an error that says how to fix
it: replace the snapshot, its meta and the log together, or delete the log
so the snapshot stands alone.

`Save`'s sequence with a log attached, every step of which is load-bearing:

```
rotate to a new segment → write the snapshot → write the meta → delete the retired segment
```

Rotating first means writes landing during the snapshot go to the new
segment, so deleting the old one cannot lose them. The meta is written
before the delete, never after: with the delete first, a crash in between
leaves a meta naming a run id no remaining segment has, and the engine
refuses to start on a set of files that is perfectly consistent. A crash
anywhere in the sequence leaves the log covering *more* history than the
snapshot needs, never less.

The meta lives beside the snapshot rather than inside it so the snapshot
format stays at v1 and every file already in the field still loads. The cost
is the case it cannot check: a snapshot written before meta files existed,
or restored without its companion, is accepted, replayed, and reported
(`LegacyMeta`, logged as a warning) rather than refused — refusing would
break every engine that predates the file.

### Health

```go
func (e *Engine) WALEnabled() bool
func (e *Engine) WALHealthy() bool
```

A write or `fsync` that fails marks the log unhealthy, and **it never
becomes healthy again**. On Linux a failed `fsync` may already have dropped
the dirty pages it could not write: the data is gone, the next call has
nothing left to fail on, and a retry returning success proves nothing about
the write that failed. For the same reason the error is sticky — every write
from then on is refused rather than acknowledged on the strength of an
`fsync` that cannot speak for the ones before it.

The process does not halt. What is in memory is still correct and still
answers every read, so it keeps serving; it just stops advertising. The
first fault fires `Options.OnWALFault` once, from its own goroutine, which
the server uses to flip the [health service](#flags) to `NOT_SERVING`, and
`wal_healthy` goes false in [Stats](#stats-rpc) for anything polling that
instead.

### Opening an engine

```go
func Open(opts Options) (*Engine, WALRecovery, error)

type Options struct {
    Dims, Capacity int
    Snapshot       string  // empty disables persistence
    WAL            string  // segment base path; empty disables logging; requires Snapshot
    OnWALFault     func()
}
```

Load, then replay, then attach:

1. **Load** the snapshot, or build an empty engine if there is no path or no
   file yet. A snapshot that exists but is corrupt refuses to start, for the
   reason in [Load](#load).
2. **Replay** the log onto it, after checking the run id against the meta.
3. **Attach** to the highest-numbered existing segment, or create segment 1.
   A brand-new segment takes the snapshot's run id rather than a fresh one,
   so the acceptance rule still holds on the next restart — minting a new id
   here would make the engine refuse to start on files it just wrote itself.

`Close` flushes and fsyncs the current segment and leaves the engine
unusable. It does **not** retire anything; callers that want the log retired
rather than merely flushed call `Save` first.

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
                        kernel_name: string; fast_int8: bool; goarch: string;
                        wal_enabled: bool; wal_healthy: bool; }

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

A durability failure *is* an error, and it stops the batch: the handler
returns `codes.Internal` naming the ID it got to and how many of the
requested IDs were deleted before it, so a caller knows exactly where to
resume. Deletions already applied stay applied in memory — as with
[Insert](#insert-rpc), batches here are not transactional.

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

`wal_enabled` and `wal_healthy` are the readiness signal in report form: a
pod whose log has faulted is still answering searches correctly, so liveness
says nothing, and this is what tells an operator to stop sending it writes.
The server also drives the [gRPC health service](#flags) off the same state.

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
| `-health-addr` | `:50052` | `grpc.health.v1.Health` listen address; empty disables the health service |
| `-dims` | `768` | vector dimension; ignored if a snapshot is loaded (the snapshot's own `dims` wins) |
| `-capacity` | `100000` | max vectors; memory for this is reserved eagerly at boot |
| `-snapshot` | `""` | snapshot file path; empty disables persistence entirely |
| `-snapshot-interval` | `0` | periodic auto-snapshot interval; `0` disables. Requires `-snapshot` to be set — the server refuses to start otherwise |
| `-wal` | `""` | write-ahead log base path. Empty means `<snapshot>.wal`, so **the log is on wherever `-snapshot` is**; `off` disables it; anything else is used as given. Requires `-snapshot` — nothing retires segments without snapshots |

The health service gets its own listener rather than sharing `-addr`,
because the data server is built with `grpc.ForceServerCodec` and that codec
would panic on a protobuf health response — see
[`DECISIONS.md`](DECISIONS.md#decision-the-health-service-listens-on-its-own-port).
It reports status for the empty service name and for `mindb.VectorService`,
and flips both to `NOT_SERVING` when the log faults or the server starts
draining. That is the endpoint a Kubernetes readiness probe should point at.

### Boot sequence

1. Logs which kernel is live (`name`, `fast_int8`, `goarch`) — the first
   thing to know when a deployment's search latency looks wrong.
2. Resolves `-wal` against `-snapshot` (see the flags table) and builds the
   health server, which starts out `SERVING`. It is built *before* the
   engine, because the fault callback closes over it.
3. `core.Open` — load, replay, attach, as described in [Opening an
   engine](#opening-an-engine). A corrupt snapshot, a log belonging to a
   different snapshot, or a dimension change all refuse to start here rather
   than booting a database that looks fine and isn't.
4. Logs `dims`, `capacity`, loaded count, estimated RAM, and what recovery
   found: how many segments and records were replayed, plus warnings for a
   truncated log, a snapshot with no `.meta` beside it, a
   `-snapshot-interval` of `0` (nothing retires segments until shutdown),
   and a disabled log.
5. Registers the gRPC server **with `grpc.ForceServerCodec(flatbuffers.FlatbuffersCodec{})`** — mandatory, since handlers return `*flatbuffers.Builder` rather than a type gRPC's default codec understands; omitting this fails at first request, not at startup.
6. Starts the health service on `-health-addr`, on a second listener with
   the default codec.
7. Starts a background periodic-snapshot goroutine if `-snapshot-interval`
   is set.
8. Serves on `-addr` in a goroutine, then blocks on either a serve error or
   an OS interrupt/`SIGTERM` signal.

### Shutdown sequence

On `SIGINT`/`SIGTERM`:

1. `healthSrv.Shutdown()` — every service goes `NOT_SERVING` *before* the
   drain, so the orchestrator stops routing here while in-flight requests
   finish rather than after.
2. Stop the periodic-snapshot goroutine.
3. `GracefulStop()` on the data server, then on the health server: finish
   in-flight RPCs, refuse new ones.
4. One final `Save` if a snapshot path is configured — deliberately ordered
   after `GracefulStop` so an in-flight write RPC can't race the final
   snapshot and get missed. This is also what retires the last log segment,
   which is why it comes before the close rather than after.
5. `engine.Close()`, which flushes and fsyncs whatever the log still holds.

### Version

`main.version` is a package-level `var` defaulting to `"dev"`, stamped at link
time with `-ldflags "-X main.version=..."`. It is the first thing the server
logs, before anything can fail:

```
mindb v0.1.0 (go1.25.1, linux/arm64)
```

A plain `go build` leaves it at `dev`, which is the honest answer rather than a
placeholder version number: an unstamped binary came from somebody's working
tree and its git state is unknown. `make image` passes `git describe`; the
release workflow passes the tag.

### Container image

`build/Dockerfile` — a two-stage build producing a static binary on
`gcr.io/distroless/static-debian12:nonroot`, about 5 MB per architecture.

The builder stage is pinned to `--platform=$BUILDPLATFORM` and reads
`TARGETARCH`, so Go cross-compiles rather than running an emulated toolchain
under QEMU. This is only free because `CGO_ENABLED=0`, which is also what makes
the static base possible. QEMU is still registered in CI, but only to assemble
the manifest list.

What the image decides, and what it deliberately leaves to the operator:

| | |
|---|---|
| `ENTRYPOINT` | `/mindb-server`, with **no `CMD`** — every flag is a deployment decision, and `-snapshot` most of all: without it the server is memory-only and the log is off |
| `USER` | `nonroot:nonroot`, uid/gid 65532 from the base image |
| `WORKDIR` | `/data`, and **not** a `VOLUME` — that creates anonymous volumes under `docker run` and is ignored by Kubernetes. The mount has to be writable by uid 65532 (`securityContext.fsGroup`) |
| `EXPOSE` | `50051` data, `50052` health |
| build args | `GO_VERSION` (default `1.25`) and `VERSION` (default `dev`, stamped as above) |

There is no shell in the image, so `kubectl exec` into a wedged pod gets you
nothing; debugging goes through the health service, [`Stats`](#stats-rpc) and
the logs. `:debug-nonroot` is the escape hatch, and it is a one-word change.

`make image` builds for the host; `make image-multi` builds
`linux/amd64,linux/arm64`. CI builds both on every push — a Dockerfile only
exercised at release time breaks at release time — and pushes to GHCR only for
`v*` tags. See [`DECISIONS.md`](DECISIONS.md#packaging).

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
- **`TestHardExitKeepsEveryAcknowledgedWrite`** and
  **`TestKilledMidWriteKeepsEveryAcknowledgedWrite`**
  (`pkg/core/wal_test.go`) — a child process inserts, prints an ID only
  *after* `Insert` returns, and is killed outright; the parent reopens and
  demands every printed ID back. This is the durability claim itself, tested
  against a real `os.Exit` and a real kill rather than a simulated one.
- **`TestTornTailAtEveryOffset`** (`pkg/core/wal_test.go`) — truncate the
  log at every byte offset from 0 to its full length and assert the engine
  opens each time, with a prefix of the writes and never a torn one.
- **`TestSilentCorruptionIsCaughtByTheChecksum`** (`pkg/core/wal_test.go`) —
  flip a bit in the last float of a vector, which decodes perfectly well.
  Only the CRC can catch that one, which is the point: the test fails if the
  checksum is computed and not compared.
- **`TestLogOrderMatchesApplyOrder`** (`pkg/core/wal_test.go`) — many
  writers contend over a few IDs with jittered stalls injected between
  applying and appending; the recovered engine must agree with the live one.
  This is the test that fails when the append moves outside `writeMu`.
- **`TestOlderSnapshotWithNewerLogRefusesToStart`** (`pkg/core/wal_test.go`)
  — restore an older snapshot next to a live log and assert the engine
  refuses to start, that the error names the log path, and that deleting the
  log fixes it.
