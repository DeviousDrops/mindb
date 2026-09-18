package core

import (
	"math/rand"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/typicallhavok/mindb/pkg/math"
)

// normalized returns what the engine stores for v: Insert normalizes in place,
// so the stored vector is v/|v|, never v.
func normalized(v []float32) []float32 {
	n := append([]float32(nil), v...)
	math.Normalize(n)
	return n
}

// mustDelete deletes id and fails the test if the deletion could not be made
// durable. It returns whether the id was there to begin with, which several
// tests assert on.
func mustDelete(t *testing.T, e *Engine, id string) bool {
	t.Helper()
	existed, err := e.Delete(id)
	if err != nil {
		t.Fatalf("delete %s: %v", id, err)
	}
	return existed
}

func mustInsert(t *testing.T, e *Engine, id string, vec []float32, payload []byte) {
	t.Helper()
	if err := e.Insert(id, vec, payload); err != nil {
		t.Fatalf("insert %s: %v", id, err)
	}
}

// ids extracts the id of each record, which is what most of these tests assert
// on: Get's contract is about which records come back and in what order.
func ids(recs []Record) []string {
	out := make([]string, len(recs))
	for i, r := range recs {
		out[i] = r.ID
	}
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestGetReturnsStoredVector(t *testing.T) {
	e, err := New(8, 4)
	if err != nil {
		t.Fatal(err)
	}
	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	mustInsert(t, e, "a", vec, []byte("payload-a"))

	got := e.Get([]string{"a"})
	if len(got) != 1 {
		t.Fatalf("got %d records, want 1", len(got))
	}
	if got[0].ID != "a" {
		t.Errorf("id = %q, want a", got[0].ID)
	}
	if string(got[0].Payload) != "payload-a" {
		t.Errorf("payload = %q, want payload-a", got[0].Payload)
	}

	// Exact equality, not an epsilon: Get reads the stored float32 vector
	// directly. It does not reconstruct from the int8 codes, so there is no
	// quantization error to tolerate.
	want := normalized(vec)
	for i := range want {
		if got[0].Vector[i] != want[i] {
			t.Fatalf("vector[%d] = %v, want %v (exact)", i, got[0].Vector[i], want[i])
		}
	}
}

// TestGetOmitsMissingIDs pins the batch contract: a missing id is absent from
// the response rather than an error, so one bad id cannot fail the other
// ninety-nine.
func TestGetOmitsMissingIDs(t *testing.T) {
	e, err := New(4, 8)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0}, nil)
	mustInsert(t, e, "c", []float32{0, 0, 1, 0}, nil)

	got := e.Get([]string{"a", "missing", "c"})
	if want := []string{"a", "c"}; !equalStrings(ids(got), want) {
		t.Errorf("ids = %v, want %v", ids(got), want)
	}
}

// TestGetPreservesRequestOrder: the caller matches responses positionally only
// if order holds, so it is part of the contract rather than an accident of map
// iteration.
func TestGetPreservesRequestOrder(t *testing.T) {
	e, err := New(4, 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"a", "b", "c"} {
		mustInsert(t, e, id, []float32{1, 0, 0, 0}, nil)
	}

	got := e.Get([]string{"c", "a", "b"})
	if want := []string{"c", "a", "b"}; !equalStrings(ids(got), want) {
		t.Errorf("ids = %v, want %v", ids(got), want)
	}
}

func TestGetReturnsNewestAfterOverwrite(t *testing.T) {
	e, err := New(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0}, []byte("first"))
	mustInsert(t, e, "a", []float32{0, 1, 0, 0}, []byte("second"))

	got := e.Get([]string{"a"})
	if len(got) != 1 {
		t.Fatalf("got %d records, want 1", len(got))
	}
	if string(got[0].Payload) != "second" {
		t.Errorf("payload = %q, want second", got[0].Payload)
	}
	if want := normalized([]float32{0, 1, 0, 0}); got[0].Vector[1] != want[1] {
		t.Errorf("vector = %v, want %v", got[0].Vector, want)
	}
}

func TestGetOmitsDeletedID(t *testing.T) {
	e, err := New(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0}, nil)
	mustInsert(t, e, "b", []float32{0, 1, 0, 0}, nil)
	if !mustDelete(t, e, "a") {
		t.Fatal("delete reported a as absent")
	}

	if got := e.Get([]string{"a", "b"}); !equalStrings(ids(got), []string{"b"}) {
		t.Errorf("ids = %v, want [b]", ids(got))
	}
}

// TestGetAfterDeletedSlotIsReused guards the free list: "a" is deleted and its
// slot handed to "b", so a Get for "a" that consulted slots rather than the id
// map would return b's data under a's name.
func TestGetAfterDeletedSlotIsReused(t *testing.T) {
	e, err := New(4, 1)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0}, []byte("a-data"))
	mustDelete(t, e, "a")
	mustInsert(t, e, "b", []float32{0, 1, 0, 0}, []byte("b-data"))

	if got := e.Get([]string{"a"}); len(got) != 0 {
		t.Errorf("get(a) = %v, want nothing after its slot was reused", ids(got))
	}
	got := e.Get([]string{"b"})
	if len(got) != 1 || string(got[0].Payload) != "b-data" {
		t.Errorf("get(b) = %v, want one record with b-data", got)
	}
}

func TestGetEmptyAndNilRequests(t *testing.T) {
	e, err := New(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0}, nil)

	if got := e.Get(nil); len(got) != 0 {
		t.Errorf("get(nil) returned %d records, want 0", len(got))
	}
	if got := e.Get([]string{}); len(got) != 0 {
		t.Errorf("get([]) returned %d records, want 0", len(got))
	}
	if got := e.Get([]string{""}); len(got) != 0 {
		t.Errorf("get(empty id) returned %d records, want 0", len(got))
	}
}

// TestGetCopiesVectorAndPayload is the one that matters for safety. The slab is
// mutable and slots are recycled, so handing out a sub-slice would let a later
// Insert rewrite a response the caller still holds.
func TestGetCopiesVectorAndPayload(t *testing.T) {
	e, err := New(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0}, []byte("original"))

	got := e.Get([]string{"a"})[0]
	before := append([]float32(nil), got.Vector...)

	// Scribble on what Get handed back.
	for i := range got.Vector {
		got.Vector[i] = 99
	}
	for i := range got.Payload {
		got.Payload[i] = 'X'
	}

	again := e.Get([]string{"a"})[0]
	for i := range before {
		if again.Vector[i] != before[i] {
			t.Fatalf("vector[%d] = %v after caller mutation, want %v; Get aliased the slab",
				i, again.Vector[i], before[i])
		}
	}
	if string(again.Payload) != "original" {
		t.Errorf("payload = %q after caller mutation, want original", again.Payload)
	}
}

// TestGetAfterRestart is the "rebuilt on snapshot load" case: the id map is
// rebuilt by Load, so a reloaded engine must answer Get identically.
func TestGetAfterRestart(t *testing.T) {
	const dims = 16
	r := rand.New(rand.NewSource(3))

	e, err := New(dims, 32)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]float32{}
	for _, id := range []string{"a", "b", "c", "d"} {
		v := randVec(r, dims)
		mustInsert(t, e, id, v, []byte(id+"-payload"))
		want[id] = normalized(v)
	}
	mustDelete(t, e, "c")

	path := filepath.Join(t.TempDir(), "snap.mindb")
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	reloaded, err := Load(path, 32)
	if err != nil {
		t.Fatal(err)
	}

	got := reloaded.Get([]string{"a", "b", "c", "d"})
	if !equalStrings(ids(got), []string{"a", "b", "d"}) {
		t.Fatalf("ids = %v, want [a b d]; c was deleted before the snapshot", ids(got))
	}
	for _, rec := range got {
		if string(rec.Payload) != rec.ID+"-payload" {
			t.Errorf("%s: payload = %q", rec.ID, rec.Payload)
		}
		for i, v := range want[rec.ID] {
			if rec.Vector[i] != v {
				t.Fatalf("%s: vector[%d] = %v, want %v; a snapshot round trip must be exact",
					rec.ID, i, rec.Vector[i], v)
			}
		}
	}
}

// TestGetConcurrentWithInsert has no assertions beyond "does not crash or
// tear": its value is under -race, which CI runs on amd64. It is not meaningful
// on a machine that cannot build the race detector.
func TestGetConcurrentWithInsert(t *testing.T) {
	const dims = 32
	e, err := New(dims, 256)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(5))
	for i := 0; i < 64; i++ {
		mustInsert(t, e, string(rune('a'+i%26))+string(rune('a'+i/26)), randVec(r, dims), []byte("p"))
	}

	var writer, readers sync.WaitGroup
	stop := make(chan struct{})

	writer.Add(1)
	go func() {
		defer writer.Done()
		w := rand.New(rand.NewSource(6))
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			id := string(rune('a'+i%26)) + string(rune('a'+(i/26)%26))
			_ = e.Insert(id, randVec(w, dims), []byte("q"))
			if i%3 == 0 {
				_, _ = e.Delete(id)
			}
		}
	}()

	for g := 0; g < 4; g++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for i := 0; i < 2000; i++ {
				for _, rec := range e.Get([]string{"aa", "ba", "ca", "za"}) {
					if len(rec.Vector) != dims {
						t.Errorf("torn record: %d dims, want %d", len(rec.Vector), dims)
						return
					}
				}
			}
		}()
	}

	// Readers run a fixed number of iterations; the writer runs until they are
	// done, so every Get overlaps live mutation.
	readers.Wait()
	close(stop)
	writer.Wait()
}

func TestStatsReportsEngineState(t *testing.T) {
	const (
		dims     = 8
		capacity = 10
	)
	e, err := New(dims, capacity)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0, 0, 0, 0, 0}, []byte("12345"))
	mustInsert(t, e, "b", []float32{0, 1, 0, 0, 0, 0, 0, 0}, []byte("123"))

	s := e.Stats()
	if s.Count != 2 {
		t.Errorf("Count = %d, want 2", s.Count)
	}
	if s.Capacity != capacity {
		t.Errorf("Capacity = %d, want %d", s.Capacity, capacity)
	}
	if s.Dims != dims {
		t.Errorf("Dims = %d, want %d", s.Dims, dims)
	}
	if want := int64(capacity) * (dims*5 + 8); s.MemoryBytes != want {
		t.Errorf("MemoryBytes = %d, want %d", s.MemoryBytes, want)
	}
	if s.PayloadBytes != 8 {
		t.Errorf("PayloadBytes = %d, want 8", s.PayloadBytes)
	}
	if s.KernelName != math.KernelName() {
		t.Errorf("KernelName = %q, want %q", s.KernelName, math.KernelName())
	}
	if s.FastInt8 != math.HasFastInt8() {
		t.Errorf("FastInt8 = %v, want %v", s.FastInt8, math.HasFastInt8())
	}
	if s.GoArch != runtime.GOARCH {
		t.Errorf("GoArch = %q, want %q", s.GoArch, runtime.GOARCH)
	}
}

// TestStatsPayloadBytesTracksMutations: payload bytes are accumulated
// incrementally rather than summed on demand, so every path that changes a
// payload has to maintain the counter. Overwrite and delete are where that goes
// wrong.
func TestStatsPayloadBytesTracksMutations(t *testing.T) {
	e, err := New(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	vec := []float32{1, 0, 0, 0}

	mustInsert(t, e, "a", vec, []byte("12345"))
	if got := e.Stats().PayloadBytes; got != 5 {
		t.Fatalf("after insert: PayloadBytes = %d, want 5", got)
	}

	mustInsert(t, e, "a", vec, []byte("1"))
	if got := e.Stats().PayloadBytes; got != 1 {
		t.Fatalf("after shrinking overwrite: PayloadBytes = %d, want 1", got)
	}

	mustInsert(t, e, "a", vec, []byte("1234567"))
	if got := e.Stats().PayloadBytes; got != 7 {
		t.Fatalf("after growing overwrite: PayloadBytes = %d, want 7", got)
	}

	mustInsert(t, e, "a", vec, nil)
	if got := e.Stats().PayloadBytes; got != 0 {
		t.Fatalf("after overwrite with no payload: PayloadBytes = %d, want 0", got)
	}

	mustInsert(t, e, "b", vec, []byte("xyz"))
	mustDelete(t, e, "b")
	if got := e.Stats().PayloadBytes; got != 0 {
		t.Fatalf("after delete: PayloadBytes = %d, want 0", got)
	}
}

func TestStatsCountAfterDelete(t *testing.T) {
	e, err := New(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "a", []float32{1, 0, 0, 0}, nil)
	mustInsert(t, e, "b", []float32{0, 1, 0, 0}, nil)
	mustDelete(t, e, "a")

	if got := e.Stats().Count; got != 1 {
		t.Errorf("Count = %d, want 1", got)
	}
}
