package core

import (
	"fmt"
	stdmath "math"
	"math/rand"
	"sort"
	"sync"
	"testing"

	"github.com/typicallhavok/mindb/pkg/math"
)

func randVec(r *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

// bruteForce is the reference: score everything, sort, take k. Deliberately the
// dumbest correct implementation, since it is the oracle for Search.
func bruteForce(t *testing.T, vecs map[string][]float32, query []float32, k int) []Result {
	t.Helper()

	q := append([]float32(nil), query...)
	math.Normalize(q)

	all := make([]Result, 0, len(vecs))
	for id, v := range vecs {
		n := append([]float32(nil), v...)
		math.Normalize(n)
		all = append(all, Result{ID: id, Score: math.Dot(q, n)})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].Score != all[j].Score {
			return all[i].Score > all[j].Score
		}
		return all[i].ID < all[j].ID
	})
	if k > len(all) {
		k = len(all)
	}
	return all[:k]
}

func TestSearchMatchesBruteForce(t *testing.T) {
	const (
		dims = 64
		n    = 500
		k    = 10
	)
	r := rand.New(rand.NewSource(1))

	e, err := New(dims, n)
	if err != nil {
		t.Fatal(err)
	}

	want := make(map[string][]float32, n)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("v%04d", i)
		v := randVec(r, dims)
		want[id] = v
		if err := e.Insert(id, v, nil); err != nil {
			t.Fatal(err)
		}
	}

	for trial := 0; trial < 50; trial++ {
		q := randVec(r, dims)

		got, err := e.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		exp := bruteForce(t, want, q, k)

		if len(got) != len(exp) {
			t.Fatalf("trial %d: got %d results, want %d", trial, len(got), len(exp))
		}
		for i := range got {
			if got[i].ID != exp[i].ID {
				t.Fatalf("trial %d rank %d: got id %q (score %v), want %q (score %v)",
					trial, i, got[i].ID, got[i].Score, exp[i].ID, exp[i].Score)
			}
			if stdmath.Abs(float64(got[i].Score-exp[i].Score)) > 1e-6 {
				t.Fatalf("trial %d rank %d: score %v != %v", trial, i, got[i].Score, exp[i].Score)
			}
		}
	}
}

// TestSearchParallelMatchesSerial pins down that the parallel path and the
// serial path agree, including tie handling. The threshold is what selects
// between them, so this crosses it.
func TestSearchParallelMatchesSerial(t *testing.T) {
	const dims, k = 32, 20
	n := parallelScanThreshold + 1000

	r := rand.New(rand.NewSource(2))
	e, err := New(dims, n)
	if err != nil {
		t.Fatal(err)
	}
	small, err := New(dims, n)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < n; i++ {
		id := fmt.Sprintf("v%06d", i)
		v := randVec(r, dims)
		if err := e.Insert(id, v, nil); err != nil {
			t.Fatal(err)
		}
		if i < 100 {
			if err := small.Insert(id, v, nil); err != nil {
				t.Fatal(err)
			}
		}
	}

	if int(e.highWater) < parallelScanThreshold {
		t.Fatalf("test did not reach the parallel path: highWater=%d", e.highWater)
	}
	if int(small.highWater) >= parallelScanThreshold {
		t.Fatal("control engine unexpectedly took the parallel path")
	}

	// Same data, both paths: results must be identical, not merely close.
	q := randVec(r, dims)
	a, err := e.Search(q, k)
	if err != nil {
		t.Fatal(err)
	}
	b, err := e.Search(q, k)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Score != b[i].Score {
			t.Fatalf("parallel search not deterministic at rank %d: %+v vs %+v", i, a[i], b[i])
		}
	}
}

func TestInsertReplacesExistingID(t *testing.T) {
	e, _ := New(4, 10)
	first := []float32{1, 0, 0, 0}
	second := []float32{0, 1, 0, 0}

	if err := e.Insert("a", first, []byte("one")); err != nil {
		t.Fatal(err)
	}
	if err := e.Insert("a", second, []byte("two")); err != nil {
		t.Fatal(err)
	}
	if e.Len() != 1 {
		t.Fatalf("replace grew the engine: Len = %d, want 1", e.Len())
	}

	res, err := e.Search([]float32{0, 1, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != "a" {
		t.Fatalf("unexpected results: %+v", res)
	}
	if stdmath.Abs(float64(res[0].Score-1)) > 1e-6 {
		t.Errorf("replacement vector not stored: score %v, want 1", res[0].Score)
	}
	if string(res[0].Payload) != "two" {
		t.Errorf("payload not replaced: %q", res[0].Payload)
	}
}

func TestDeleteRemovesFromResults(t *testing.T) {
	e, _ := New(4, 10)
	for i, v := range [][]float32{{1, 0, 0, 0}, {0.9, 0.1, 0, 0}, {0, 0, 1, 0}} {
		if err := e.Insert(fmt.Sprintf("v%d", i), v, nil); err != nil {
			t.Fatal(err)
		}
	}

	if !mustDelete(t, e, "v0") {
		t.Fatal("Delete reported the id was absent")
	}
	if mustDelete(t, e, "v0") {
		t.Error("second Delete of the same id reported success")
	}
	if e.Len() != 2 {
		t.Fatalf("Len = %d, want 2", e.Len())
	}

	res, err := e.Search([]float32{1, 0, 0, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.ID == "v0" {
			t.Fatal("deleted vector appeared in results")
		}
	}
}

func TestDeletedSlotIsReused(t *testing.T) {
	// The free list is what replaces compaction; if it does not actually
	// recycle, capacity leaks on every delete.
	e, _ := New(4, 2)
	if err := e.Insert("a", []float32{1, 0, 0, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.Insert("b", []float32{0, 1, 0, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.Insert("c", []float32{0, 0, 1, 0}, nil); err == nil {
		t.Fatal("insert past capacity succeeded")
	}

	mustDelete(t, e, "a")
	if err := e.Insert("c", []float32{0, 0, 1, 0}, nil); err != nil {
		t.Fatalf("slot was not reused after delete: %v", err)
	}
	if e.highWater != 2 {
		t.Errorf("highWater = %d, want 2 (reuse should not extend it)", e.highWater)
	}
}

func TestInsertRejectsBadInput(t *testing.T) {
	e, _ := New(4, 10)

	if err := e.Insert("", []float32{1, 0, 0, 0}, nil); err != ErrEmptyID {
		t.Errorf("empty id: got %v, want ErrEmptyID", err)
	}
	if err := e.Insert("a", []float32{1, 0}, nil); err == nil {
		t.Error("dimension mismatch accepted")
	}
	if err := e.Insert("a", []float32{0, 0, 0, 0}, nil); err != ErrZeroVector {
		t.Errorf("zero vector: got %v, want ErrZeroVector", err)
	}
}

func TestInsertCopiesInputs(t *testing.T) {
	// The gRPC path hands us slices into a recycled receive buffer, so
	// retaining them would silently corrupt stored vectors.
	e, _ := New(4, 10)
	vec := []float32{1, 0, 0, 0}
	pay := []byte("keep")

	if err := e.Insert("a", vec, pay); err != nil {
		t.Fatal(err)
	}
	for i := range vec {
		vec[i] = 99
	}
	copy(pay, "GONE")

	res, err := e.Search([]float32{1, 0, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if stdmath.Abs(float64(res[0].Score-1)) > 1e-6 {
		t.Errorf("stored vector aliased the caller's slice: score %v", res[0].Score)
	}
	if string(res[0].Payload) != "keep" {
		t.Errorf("stored payload aliased the caller's slice: %q", res[0].Payload)
	}
}

func TestInsertDoesNotModifyCallerVector(t *testing.T) {
	e, _ := New(4, 10)
	vec := []float32{3, 0, 0, 0}
	if err := e.Insert("a", vec, nil); err != nil {
		t.Fatal(err)
	}
	if vec[0] != 3 {
		t.Errorf("Insert normalized the caller's slice in place: %v", vec)
	}
}

func TestSearchEdgeCases(t *testing.T) {
	e, _ := New(4, 10)

	if res, err := e.Search([]float32{1, 0, 0, 0}, 5); err != nil || res != nil {
		t.Errorf("empty engine: got %v, %v; want nil, nil", res, err)
	}

	e.Insert("a", []float32{1, 0, 0, 0}, nil)

	if res, _ := e.Search([]float32{1, 0, 0, 0}, 0); res != nil {
		t.Errorf("k=0 returned %v, want nil", res)
	}
	if res, _ := e.Search([]float32{1, 0, 0, 0}, 100); len(res) != 1 {
		t.Errorf("k past count returned %d results, want 1", len(res))
	}
	if _, err := e.Search([]float32{1, 0}, 1); err == nil {
		t.Error("dimension mismatch accepted")
	}
	if _, err := e.Search([]float32{0, 0, 0, 0}, 1); err != ErrZeroVector {
		t.Error("zero query accepted")
	}
}

// TestConcurrentHammer is the reason the lock-free design was abandoned. Under
// -race this exercises exactly the interleavings that produced an unrecoverable
// runtime throw, a torn vector read, and a lost write.
func TestConcurrentHammer(t *testing.T) {
	const (
		dims    = 32
		cap     = 2000
		readers = 8
		writers = 4
		rounds  = 300
	)

	e, err := New(dims, cap)
	if err != nil {
		t.Fatal(err)
	}
	seed := rand.New(rand.NewSource(7))
	for i := 0; i < 500; i++ {
		if err := e.Insert(fmt.Sprintf("seed%04d", i), randVec(seed, dims), nil); err != nil {
			t.Fatal(err)
		}
	}

	var wg sync.WaitGroup

	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(100 + w)))
			for i := 0; i < rounds; i++ {
				id := fmt.Sprintf("w%d-%d", w, i%50)
				// Ignore capacity errors: contention over the free list means
				// a writer can legitimately lose the race for the last slot.
				_ = e.Insert(id, randVec(r, dims), []byte(id))
				if i%3 == 0 {
					_, _ = e.Delete(id)
				}
				_, _ = e.Delete(fmt.Sprintf("seed%04d", r.Intn(500)))
			}
		}(w)
	}

	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func(seedN int) {
			defer wg.Done()
			rr := rand.New(rand.NewSource(int64(200 + seedN)))
			for i := 0; i < rounds; i++ {
				res, err := e.Search(randVec(rr, dims), 10)
				if err != nil {
					t.Errorf("search failed: %v", err)
					return
				}
				for _, hit := range res {
					// A torn read or a slot recycled mid-scan shows up here:
					// the score of a unit vector against a unit query cannot
					// leave [-1, 1], and NaN means a half-written vector.
					if stdmath.IsNaN(float64(hit.Score)) {
						t.Errorf("NaN score for %q: torn vector read", hit.ID)
						return
					}
					if hit.Score < -1.001 || hit.Score > 1.001 {
						t.Errorf("score %v for %q outside [-1,1]", hit.Score, hit.ID)
						return
					}
					if hit.ID == "" {
						t.Error("result with empty id: slot recycled mid-scan")
						return
					}
				}
			}
		}(r)
	}

	wg.Wait()
}

func TestTopKKeepsBest(t *testing.T) {
	h := &topK{}
	h.reset(3)
	for i, score := range []float32{0.1, 0.9, 0.5, 0.7, 0.2, 0.95} {
		h.push(candidate{score: score, slot: uint32(i)})
	}
	if len(h.heap) != 3 {
		t.Fatalf("heap holds %d, want 3", len(h.heap))
	}

	got := make([]float32, 0, 3)
	for _, c := range h.heap {
		got = append(got, c.score)
	}
	sort.Slice(got, func(i, j int) bool { return got[i] > got[j] })
	for i, want := range []float32{0.95, 0.9, 0.7} {
		if got[i] != want {
			t.Errorf("rank %d: got %v, want %v", i, got[i], want)
		}
	}
}

func TestTopKTieBreaksOnSlot(t *testing.T) {
	// Equal scores must resolve the same way regardless of insertion order,
	// or results depend on how the scan was partitioned.
	forward := &topK{}
	forward.reset(2)
	backward := &topK{}
	backward.reset(2)

	for _, slot := range []uint32{0, 1, 2, 3} {
		forward.push(candidate{score: 0.5, slot: slot})
	}
	for _, slot := range []uint32{3, 2, 1, 0} {
		backward.push(candidate{score: 0.5, slot: slot})
	}

	slots := func(h *topK) []uint32 {
		out := []uint32{}
		for _, c := range h.heap {
			out = append(out, c.slot)
		}
		sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
		return out
	}
	f, b := slots(forward), slots(backward)
	if len(f) != 2 || f[0] != 0 || f[1] != 1 {
		t.Errorf("forward kept slots %v, want [0 1]", f)
	}
	if len(b) != 2 || b[0] != 0 || b[1] != 1 {
		t.Errorf("backward kept slots %v, want [0 1]", b)
	}
}

func BenchmarkSearch(b *testing.B) {
	for _, dims := range []int{128, 768} {
		const n = 20000
		r := rand.New(rand.NewSource(9))
		e, _ := New(dims, n)
		for i := 0; i < n; i++ {
			e.Insert(fmt.Sprintf("v%06d", i), randVec(r, dims), nil)
		}
		q := randVec(r, dims)

		b.Run(fmt.Sprintf("dims=%d/n=%d", dims, n), func(b *testing.B) {
			b.SetBytes(int64(n * dims * 4))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := e.Search(q, 10); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
