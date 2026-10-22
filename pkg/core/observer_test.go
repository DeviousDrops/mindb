package core

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"
)

// recorder is an Observer that keeps every event it is handed.
type recorder struct {
	mu        sync.Mutex
	applies   int
	syncs     []WALSync
	searches  []SearchEvent
	snapshots []SnapshotEvent
}

func (r *recorder) WALSync(e WALSync) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.syncs = append(r.syncs, e)
}

func (r *recorder) WriteApply(time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.applies++
}

func (r *recorder) Search(e SearchEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.searches = append(r.searches, e)
}

func (r *recorder) Snapshot(e SnapshotEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.snapshots = append(r.snapshots, e)
}

// Every acknowledged write has to be counted in exactly one sync. If a record
// were dropped from the batch accounting, or counted by two leaders, the batch
// size histogram would lie about how well group commit is working — which is
// the one thing it exists to show.
func TestObserverCountsEveryRecordOnce(t *testing.T) {
	h := newWALHarness(t)
	obs := &recorder{}
	opts := h.options()
	opts.Observer = obs
	e, _, err := Open(opts)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	const writers, each = 16, 20
	var (
		wg        sync.WaitGroup
		wantBytes int64
		bytesMu   sync.Mutex
	)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(w)))
			for i := 0; i < each; i++ {
				id := fmt.Sprintf("w%d-%d", w, i)
				vec := randVec(r, walTestDims)
				if err := e.Insert(id, vec, nil); err != nil {
					t.Error(err)
					return
				}
				// The record is framed from the normalized copy, but its
				// length depends only on the id and the dimension.
				n := int64(len(appendInsert(nil, id, make([]float32, walTestDims), nil)))
				bytesMu.Lock()
				wantBytes += n
				bytesMu.Unlock()
			}
		}(w)
	}
	wg.Wait()

	obs.mu.Lock()
	var records int
	var bytes int64
	for _, s := range obs.syncs {
		if s.Err != nil {
			t.Fatalf("sync reported error: %v", s.Err)
		}
		if s.Rotation {
			t.Fatal("no Save ran, so no sync should be a rotation")
		}
		if s.Records < 1 {
			t.Fatalf("a writer-led sync made %d records durable", s.Records)
		}
		records += s.Records
		bytes += s.Bytes
	}
	nSyncs := len(obs.syncs)
	obs.mu.Unlock()

	if records != writers*each {
		t.Fatalf("records across syncs = %d, want %d", records, writers*each)
	}
	if obs.applies != writers*each {
		t.Fatalf("%d write-apply events for %d writes", obs.applies, writers*each)
	}
	if bytes != wantBytes {
		t.Fatalf("bytes across syncs = %d, want %d", bytes, wantBytes)
	}
	if nSyncs > writers*each {
		t.Fatalf("%d syncs for %d writes", nSyncs, writers*each)
	}
	t.Logf("%d writes in %d fsyncs (mean batch %.1f)", writers*each, nSyncs, float64(writers*each)/float64(nSyncs))
}

func TestObserverSnapshotAndRotation(t *testing.T) {
	h := newWALHarness(t)
	obs := &recorder{}
	opts := h.options()
	opts.Observer = obs
	e, _, err := Open(opts)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	r := rand.New(rand.NewSource(1))
	for i := 0; i < 5; i++ {
		if err := e.Insert(fmt.Sprint(i), randVec(r, walTestDims), []byte("p")); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.Save(h.snapshot); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(h.snapshot)
	if err != nil {
		t.Fatal(err)
	}

	obs.mu.Lock()
	defer obs.mu.Unlock()

	if len(obs.snapshots) != 1 {
		t.Fatalf("got %d snapshot events, want 1", len(obs.snapshots))
	}
	if s := obs.snapshots[0]; s.Err != nil || s.Bytes != info.Size() || s.Duration <= 0 {
		t.Fatalf("snapshot event %+v, file is %d bytes", s, info.Size())
	}

	last := obs.syncs[len(obs.syncs)-1]
	if !last.Rotation {
		t.Fatalf("last sync should be the rotation, got %+v", last)
	}
	// Every insert above waited for its own sync, so the rotation found
	// nothing pending.
	if last.Records != 0 || last.Bytes != 0 {
		t.Fatalf("rotation reported %d records / %d bytes pending, want 0", last.Records, last.Bytes)
	}

	segs, bytes, err := e.WALDiskUsage()
	if err != nil {
		t.Fatal(err)
	}
	if segs != 1 || bytes != walHeaderSize {
		t.Fatalf("after Save: %d segments, %d bytes; want 1 segment holding just its header (%d)", segs, bytes, walHeaderSize)
	}
}

func TestObserverSearchPaths(t *testing.T) {
	const dims, n = 32, 2000
	obs := &recorder{}
	e, _, err := Open(Options{Dims: dims, Capacity: n, Observer: obs})
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(7))
	centroids := makeCentroids(r, 20, dims)
	for i := 0; i < n; i++ {
		if err := e.Insert(fmt.Sprint(i), clusteredVec(r, centroids, 0.05), nil); err != nil {
			t.Fatal(err)
		}
	}
	q := clusteredVec(r, centroids, 0.05)

	last := func() SearchEvent {
		obs.mu.Lock()
		defer obs.mu.Unlock()
		return obs.searches[len(obs.searches)-1]
	}

	e.SetCascade(false)
	if _, err := e.Search(q, 10); err != nil {
		t.Fatal(err)
	}
	if ev := last(); ev.Path != PathScan || ev.Live != n || ev.Survivors != 0 || ev.Duration <= 0 {
		t.Fatalf("scan: %+v", ev)
	}

	// The cascade path is taken whenever it is forced on, whether or not a
	// SIMD kernel exists; only its speed depends on the kernel.
	e.SetCascade(true)
	if _, err := e.Search(q, 10); err != nil {
		t.Fatal(err)
	}
	if ev := last(); ev.Path != PathCascade || ev.Survivors < 10 || ev.Survivors > n/4 {
		t.Fatalf("cascade: %+v", ev)
	}

	original := guardFraction
	guardFraction = 0 // any survivor at all trips the guard
	defer func() { guardFraction = original }()
	if _, err := e.Search(q, 10); err != nil {
		t.Fatal(err)
	}
	if ev := last(); ev.Path != PathFallback || ev.Survivors != n {
		t.Fatalf("fallback: %+v", ev)
	}
}

// An engine built without an observer must not report to one, and must not
// pay for the clock reads either; the second part is not testable here, but
// the first guards against a nil interface being called.
func TestNoObserverIsFine(t *testing.T) {
	h := newWALHarness(t)
	e, _, err := Open(h.options())
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if err := e.Insert("a", randVec(rand.New(rand.NewSource(1)), walTestDims), nil); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Search(randVec(rand.New(rand.NewSource(2)), walTestDims), 1); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(h.snapshot); err != nil {
		t.Fatal(err)
	}
}
