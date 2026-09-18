// Package core holds MinDB's in-memory vector store.
//
// The concurrency model is one RWMutex held across the whole scan. That is a
// deliberate reversal of an earlier lock-free design; the reasoning is in
// docs/ARCHITECTURE.md and comes down to arithmetic: RLock costs ~20 ns and a
// scan costs ~6,000,000 ns, so the lock is 0.0003% of the operation, and the
// lock-free version was paying for that rounding error with a hard crash, a
// silent-corruption bug and a lost-write bug.
package core

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"

	"github.com/typicallhavok/mindb/pkg/math"
)

var (
	ErrDimensionMismatch = errors.New("mindb: vector dimension does not match engine")
	ErrCapacityExceeded  = errors.New("mindb: engine is at capacity")
	ErrZeroVector        = errors.New("mindb: cannot insert a zero vector")
	ErrEmptyID           = errors.New("mindb: vector id must not be empty")
)

// parallelScanThreshold is the number of live slots below which Search runs on
// one goroutine. Below this the fan-out and merge cost more than the scan.
const parallelScanThreshold = 8192

// Result is one search hit.
type Result struct {
	ID      string
	Score   float32
	Payload []byte
}

// Engine is an in-memory, exact k-NN vector store over unit-normalized vectors.
//
// Layout is struct-of-arrays: a scan touches only the arrays it needs, so every
// cache line pulled in is entirely useful. Vectors are normalized at insert,
// which makes cosine similarity equal the dot product — that removes a division
// and a random-access magnitude load from the hot loop, and it is a
// precondition of the cascade's error bound in stage 2.
//
// The zero value is not usable; call New.
type Engine struct {
	dims     int
	capacity int

	// writeMu serializes mutating operations. It is not mu: the fsync at the end
	// of a write happens with this held and mu released, so a write never blocks
	// a reader on disk. Its job is to make the order operations reach the log the
	// same order they reached the slab — without it two concurrent writes to one
	// id could be applied in one order and logged in the other, and recovery
	// would disagree with what was live.
	writeMu sync.Mutex
	wal     *wal

	mu         sync.RWMutex
	vectors    []float32 // capacity*dims, unit-normalized
	codes      []int8    // capacity*dims, int8 quantization of vectors
	scales     []float32 // capacity, per-vector quantization scale
	residuals  []float32 // capacity, ρ = ‖v − v̂‖; the pruning bound
	externalID []string  // capacity
	payloads   [][]byte  // capacity
	live       []bool    // capacity; slot occupied
	idMap      map[string]uint32
	free       []uint32 // recycled slots, replacing compaction entirely
	highWater  uint32   // slots at or beyond this were never used; scan stops here
	count      int

	// payloadBytes is maintained incrementally rather than summed on demand:
	// payloads are the only variable-size allocation the engine owns, and
	// walking every live slot to total them would make Stats O(capacity).
	// Every path that stores or drops a payload has to keep this honest.
	payloadBytes int64

	// useCascade selects the bound-and-refine path over a plain float32 scan.
	// Both return identical results; the cascade is only worth taking when a
	// SIMD int8 kernel is available, since pure Go int8 is slower than float32.
	useCascade bool

	heaps   sync.Pool // *topK, reused across queries
	scratch sync.Pool // *[]float32, approximate scores indexed by slot
}

// New allocates an engine for exactly capacity vectors of the given dimension.
//
// Allocation is eager: the full capacity*dims float32 array is reserved now, so
// the process either has the memory or fails here rather than under load. At
// 100k x 768 that is ~293 MiB for the vectors alone.
func New(dims, capacity int) (*Engine, error) {
	if dims <= 0 {
		return nil, fmt.Errorf("mindb: dims must be positive, got %d", dims)
	}
	if capacity <= 0 {
		return nil, fmt.Errorf("mindb: capacity must be positive, got %d", capacity)
	}

	e := &Engine{
		dims:       dims,
		capacity:   capacity,
		vectors:    make([]float32, capacity*dims),
		codes:      make([]int8, capacity*dims),
		scales:     make([]float32, capacity),
		residuals:  make([]float32, capacity),
		externalID: make([]string, capacity),
		payloads:   make([][]byte, capacity),
		live:       make([]bool, capacity),
		idMap:      make(map[string]uint32, capacity),
		useCascade: math.HasFastInt8(),
	}
	e.heaps.New = func() any { return &topK{} }
	e.scratch.New = func() any { s := make([]float32, capacity); return &s }
	return e, nil
}

// SetCascade forces the search path on or off, overriding the automatic choice.
//
// Both paths return identical results — that is the point of the design and is
// asserted by the differential test — so this exists for benchmarking and for
// operators who want to pin behaviour, not to trade accuracy for speed.
func (e *Engine) SetCascade(on bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.useCascade = on
}

// Cascade reports whether the bound-and-refine path is in use.
func (e *Engine) Cascade() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.useCascade
}

// Dims returns the vector dimension this engine was configured with.
func (e *Engine) Dims() int { return e.dims }

// Cap returns the maximum number of vectors the engine can hold.
func (e *Engine) Cap() int { return e.capacity }

// Len returns the number of live vectors.
func (e *Engine) Len() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.count
}

// Insert stores vec under id, replacing any existing vector with that id.
//
// vec and payload are copied; the caller may reuse both immediately. This is
// not optional politeness — the gRPC path hands us slices pointing into a
// FlatBuffers receive buffer that is recycled the moment the handler returns.
//
// The stored copy is unit-normalized. vec itself is left untouched.
func (e *Engine) Insert(id string, vec []float32, payload []byte) error {
	if id == "" {
		return ErrEmptyID
	}
	if len(vec) != e.dims {
		return fmt.Errorf("%w: got %d, want %d", ErrDimensionMismatch, len(vec), e.dims)
	}

	// Normalize outside the lock: it is O(dims) and touches only our own copy.
	buf := make([]float32, e.dims)
	copy(buf, vec)
	if math.Normalize(buf) == 0 {
		return ErrZeroVector
	}

	if e.wal == nil {
		return e.store(id, buf, payload)
	}

	// Encoded before the lock: it is the largest allocation on this path and
	// nothing about it depends on engine state.
	rec := appendInsert(nil, id, buf, payload)

	e.writeMu.Lock()
	// Applied before it is logged. store's only failure is ErrCapacityExceeded,
	// and taking it here means the log never carries a record that cannot be
	// replayed. See the wal type for what that ordering costs.
	if err := e.store(id, buf, payload); err != nil {
		e.writeMu.Unlock()
		return err
	}
	seq := e.wal.buffer(rec)
	e.writeMu.Unlock()

	return e.wal.syncTo(seq)
}

// store publishes an already-normalized vector under id. buf is adopted, not
// copied, so callers must not retain it; payload is copied.
//
// Split out from Insert so snapshot loading can skip normalization. Re-running
// Normalize on an already-unit vector is not the identity in float32 — the norm
// computes as 1±1e-7 and dividing by it shifts the low bits — which would make
// a snapshot round-trip return subtly different scores.
func (e *Engine) store(id string, buf []float32, payload []byte) error {
	var pay []byte
	if len(payload) > 0 {
		pay = make([]byte, len(payload))
		copy(pay, payload)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	slot, existing := e.idMap[id]
	if !existing {
		var err error
		if slot, err = e.allocSlot(); err != nil {
			return err
		}
		e.idMap[id] = slot
		e.externalID[slot] = id
		e.live[slot] = true
		e.count++
	}

	base := int(slot) * e.dims
	copy(e.vectors[base:], buf)
	e.scales[slot], e.residuals[slot] = math.Quantize(buf, e.codes[base:base+e.dims])

	// Overwriting an existing id reuses its slot, so the payload it is replacing
	// has to come off the total first.
	e.payloadBytes += int64(len(pay)) - int64(len(e.payloads[slot]))
	e.payloads[slot] = pay
	return nil
}

// allocSlot returns a usable slot index. Caller must hold the write lock.
func (e *Engine) allocSlot() (uint32, error) {
	if n := len(e.free); n > 0 {
		slot := e.free[n-1]
		e.free = e.free[:n-1]
		return slot, nil
	}
	if int(e.highWater) >= e.capacity {
		return 0, fmt.Errorf("%w: %d vectors", ErrCapacityExceeded, e.capacity)
	}
	slot := e.highWater
	e.highWater++
	return slot, nil
}

// Delete removes the vector with the given id, reporting whether it existed.
//
// The slot goes on the free list for reuse. There is no compaction: capacity is
// preallocated, so compacting would move data around inside an array that never
// changes size. A free list gets the same reuse in ten lines with no concurrency
// exposure at all.
//
// The error is non-nil only when a write-ahead log is attached and the record
// could not be made durable. It is reported rather than swallowed because a
// delete that is acknowledged and then lost resurrects the vector on the next
// restart, which is the worst failure this package can have.
func (e *Engine) Delete(id string) (bool, error) {
	if e.wal == nil {
		return e.deleteSlot(id), nil
	}

	rec := appendDelete(nil, id)

	e.writeMu.Lock()
	existed := e.deleteSlot(id)
	if !existed {
		// Nothing changed, so there is nothing to replay. Logging no-op deletes
		// would let a caller retrying a delete in a loop grow the log without
		// ever changing the state it describes.
		e.writeMu.Unlock()
		return false, nil
	}
	seq := e.wal.buffer(rec)
	e.writeMu.Unlock()

	return true, e.wal.syncTo(seq)
}

// deleteSlot removes id from the slab, reporting whether it was there.
func (e *Engine) deleteSlot(id string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()

	slot, ok := e.idMap[id]
	if !ok {
		return false
	}
	delete(e.idMap, id)
	e.live[slot] = false
	e.externalID[slot] = ""
	e.payloadBytes -= int64(len(e.payloads[slot]))
	e.payloads[slot] = nil // drop the reference so the payload can be collected
	e.free = append(e.free, slot)
	e.count--
	return true
}

// Search returns the k vectors most similar to query, best first.
//
// query is normalized internally and is not modified. Results are exact: every
// live vector is scored. Ties are broken by slot index so that output is fully
// deterministic regardless of how the scan was partitioned across workers —
// stage 2's differential test compares scores exactly and would otherwise be
// flaky under a different GOMAXPROCS.
func (e *Engine) Search(query []float32, k int) ([]Result, error) {
	if len(query) != e.dims {
		return nil, fmt.Errorf("%w: got %d, want %d", ErrDimensionMismatch, len(query), e.dims)
	}
	if k <= 0 {
		return nil, nil
	}

	q := make([]float32, e.dims)
	copy(q, query)
	if math.Normalize(q) == 0 {
		return nil, ErrZeroVector
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.count == 0 {
		return nil, nil
	}
	if k > e.count {
		k = e.count
	}

	var best *topK
	if e.useCascade {
		best, _ = e.searchCascade(q, k)
	} else {
		best = e.scan(q, k)
	}
	defer e.heaps.Put(best)

	out := make([]Result, len(best.heap))
	for i, c := range best.heap {
		out[i] = Result{
			ID: e.externalID[c.slot],
			// Payload aliases engine-owned memory rather than being copied.
			// A concurrent Delete drops the engine's reference but cannot
			// invalidate this one, so the worst case is a stale read, never a
			// use-after-free. Callers that retain it past the call must copy.
			Score:   c.score,
			Payload: e.payloads[c.slot],
		}
	}
	// Heap order is not sorted order; k is small, so this is negligible.
	// Presentation ties break on ID rather than slot because slots are
	// reassigned across a snapshot reload and IDs are not.
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ID < out[j].ID
	})
	return out, nil
}

// scan scores every live slot below highWater and returns the top k.
// Caller must hold at least the read lock.
func (e *Engine) scan(q []float32, k int) *topK {
	hw := int(e.highWater)

	workers := runtime.GOMAXPROCS(0)
	if hw < parallelScanThreshold || workers < 2 {
		h := e.takeHeap(k)
		e.scanRange(q, 0, hw, h)
		return h
	}

	chunk := (hw + workers - 1) / workers
	heaps := make([]*topK, workers)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= hw {
			break
		}
		hi := lo + chunk
		if hi > hw {
			hi = hw
		}
		heaps[w] = e.takeHeap(k)
		wg.Add(1)
		go func(lo, hi int, h *topK) {
			defer wg.Done()
			e.scanRange(q, lo, hi, h)
		}(lo, hi, heaps[w])
	}
	wg.Wait()

	merged := heaps[0]
	for _, h := range heaps[1:] {
		if h == nil {
			continue
		}
		for _, c := range h.heap {
			merged.push(c)
		}
		e.heaps.Put(h)
	}
	return merged
}

// scanRange scores slots in [lo, hi) into h. This is the hot loop; it must not
// allocate.
func (e *Engine) scanRange(q []float32, lo, hi int, h *topK) {
	dims := e.dims
	for slot := lo; slot < hi; slot++ {
		if !e.live[slot] {
			continue
		}
		base := slot * dims
		h.push(candidate{
			slot:  uint32(slot),
			score: math.Dot(q, e.vectors[base:base+dims]),
		})
	}
}

func (e *Engine) takeHeap(k int) *topK {
	h := e.heaps.Get().(*topK)
	h.reset(k)
	return h
}

type candidate struct {
	score float32
	slot  uint32
}

// worse reports whether a ranks below b under the total order used for top-k:
// higher score wins, and equal scores are broken by lower slot index. The
// tiebreak exists so results do not depend on scan partitioning.
func worse(a, b candidate) bool {
	if a.score != b.score {
		return a.score < b.score
	}
	return a.slot > b.slot
}

// topK is a bounded min-heap of size k: the weakest survivor sits at the root,
// so it is the one evicted when a better candidate arrives.
//
// This is a heap rather than a sort because sorting 100k scores is O(n log n)
// plus an allocation, which would erase every gain the rest of the design makes.
type topK struct {
	k    int
	heap []candidate
}

func (t *topK) reset(k int) {
	t.k = k
	if cap(t.heap) < k {
		t.heap = make([]candidate, 0, k)
	}
	t.heap = t.heap[:0]
}

func (t *topK) push(c candidate) {
	if len(t.heap) < t.k {
		t.heap = append(t.heap, c)
		t.siftUp(len(t.heap) - 1)
		return
	}
	// The overwhelmingly common case: one predictable, mispredict-free compare
	// that rejects the candidate outright.
	if c.score < t.heap[0].score {
		return
	}
	if !worse(t.heap[0], c) {
		return
	}
	t.heap[0] = c
	t.siftDown(0)
}

func (t *topK) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !worse(t.heap[i], t.heap[parent]) {
			break
		}
		t.heap[i], t.heap[parent] = t.heap[parent], t.heap[i]
		i = parent
	}
}

func (t *topK) siftDown(i int) {
	n := len(t.heap)
	for {
		smallest := i
		if l := 2*i + 1; l < n && worse(t.heap[l], t.heap[smallest]) {
			smallest = l
		}
		if r := 2*i + 2; r < n && worse(t.heap[r], t.heap[smallest]) {
			smallest = r
		}
		if smallest == i {
			return
		}
		t.heap[i], t.heap[smallest] = t.heap[smallest], t.heap[i]
		i = smallest
	}
}
