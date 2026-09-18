package core

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// The durability tests below all follow the same shape: write through a real
// engine, lose the process or damage the file, open it again, and compare what
// comes back against what was acknowledged. Nothing is asserted about writes
// that were still in flight, because nothing was promised about them.

const walTestDims = 8

// walHarness is a directory with a snapshot path and a log base path in it.
type walHarness struct {
	t        *testing.T
	dir      string
	snapshot string
	wal      string
	capacity int
	faults   chan struct{}
}

func newWALHarness(t *testing.T) *walHarness {
	t.Helper()
	dir := t.TempDir()
	return &walHarness{
		t:        t,
		dir:      dir,
		snapshot: filepath.Join(dir, "snap.mindb"),
		wal:      filepath.Join(dir, "snap.mindb.wal"),
		capacity: 512,
		faults:   make(chan struct{}, 1),
	}
}

func (h *walHarness) options() Options {
	return Options{
		Dims:     walTestDims,
		Capacity: h.capacity,
		Snapshot: h.snapshot,
		WAL:      h.wal,
		OnWALFault: func() {
			select {
			case h.faults <- struct{}{}:
			default:
			}
		},
	}
}

// open starts an engine over whatever is on disk, failing the test if it will
// not start. The engine is closed when the test ends, so a test that opens the
// same directory twice must close the first one itself.
func (h *walHarness) open() (*Engine, WALRecovery) {
	h.t.Helper()
	e, rec, err := Open(h.options())
	if err != nil {
		h.t.Fatalf("open: %v", err)
	}
	return e, rec
}

// walVec is a deterministic non-zero vector, distinct for every i.
func walVec(i int) []float32 {
	v := make([]float32, walTestDims)
	v[i%walTestDims] = 1
	v[(i*3+1)%walTestDims] += float32(i%17) + 1
	return v
}

// contents reads every live record out of an engine, so two engines can be
// compared on what they hold rather than on how they got there.
func contents(t *testing.T, e *Engine) map[string]Record {
	t.Helper()
	e.mu.RLock()
	ids := make([]string, 0, len(e.idMap))
	for id := range e.idMap {
		ids = append(ids, id)
	}
	e.mu.RUnlock()

	out := make(map[string]Record)
	for _, r := range e.Get(ids) {
		out[r.ID] = r
	}
	return out
}

func sameContents(t *testing.T, got, want map[string]Record) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("holds %d records, want %d", len(got), len(want))
	}
	for id, w := range want {
		g, ok := got[id]
		if !ok {
			t.Fatalf("%s is missing", id)
		}
		if string(g.Payload) != string(w.Payload) {
			t.Errorf("%s payload = %q, want %q", id, g.Payload, w.Payload)
		}
		if len(g.Vector) != len(w.Vector) {
			t.Fatalf("%s vector has %d dims, want %d", id, len(g.Vector), len(w.Vector))
		}
		for i := range w.Vector {
			// Exact, not approximate. Replay stores the logged floats as they
			// are; re-normalizing a unit vector is not the identity in float32,
			// so a drift of one bit here means replay went through Insert.
			if g.Vector[i] != w.Vector[i] {
				t.Fatalf("%s vector[%d] = %v, want %v", id, i, g.Vector[i], w.Vector[i])
			}
		}
	}
}

func TestReopenSeesEveryAcknowledgedWrite(t *testing.T) {
	h := newWALHarness(t)

	e, rec := h.open()
	if rec.Segments != 0 || rec.Records != 0 {
		t.Fatalf("fresh directory recovered %d records in %d segments", rec.Records, rec.Segments)
	}
	for i := 0; i < 50; i++ {
		mustInsert(t, e, fmt.Sprintf("k%02d", i), walVec(i), []byte(fmt.Sprintf("p%02d", i)))
	}
	mustDelete(t, e, "k07")
	want := contents(t, e)
	if err := e.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// No Save anywhere above: everything here comes back out of the log.
	reopened, rec := h.open()
	defer reopened.Close()
	if rec.Records != 51 {
		t.Errorf("replayed %d records, want 51 (50 inserts and a delete)", rec.Records)
	}
	if rec.Truncated {
		t.Error("a cleanly closed log was reported as truncated")
	}
	sameContents(t, contents(t, reopened), want)
	if _, ok := contents(t, reopened)["k07"]; ok {
		t.Error("k07 came back after being deleted")
	}
}

func TestReplayIsIdempotent(t *testing.T) {
	h := newWALHarness(t)

	e, _ := h.open()
	for i := 0; i < 20; i++ {
		mustInsert(t, e, fmt.Sprintf("k%02d", i), walVec(i), nil)
	}
	want := contents(t, e)
	e.Close()

	// The same log replayed twice, and then a third time over a snapshot that
	// already contains all of it. Insert and delete are blind writes, so every
	// one of these has to land on the same state.
	for round := 0; round < 2; round++ {
		again, _ := h.open()
		sameContents(t, contents(t, again), want)
		again.Close()
	}

	saved, _ := h.open()
	if err := saved.Save(h.snapshot); err != nil {
		t.Fatalf("save: %v", err)
	}
	saved.Close()

	last, _ := h.open()
	defer last.Close()
	sameContents(t, contents(t, last), want)
}

// recordEnds returns the file offset just past each record in a segment.
func recordEnds(t *testing.T, path string) []int64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	s := newWALScanner(f, walTestDims)
	if err := s.skipHeader(); err != nil {
		t.Fatalf("header: %v", err)
	}
	var ends []int64
	for {
		if _, ok := s.next(); !ok {
			return ends
		}
		ends = append(ends, s.off)
	}
}

// buildLog writes n inserts and returns the segment path and the harness.
func buildLog(t *testing.T, n int) (*walHarness, string) {
	t.Helper()
	h := newWALHarness(t)
	e, _ := h.open()
	for i := 0; i < n; i++ {
		mustInsert(t, e, fmt.Sprintf("k%02d", i), walVec(i), []byte(fmt.Sprintf("p%02d", i)))
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	return h, segmentName(h.wal, 1)
}

func TestTornTailAtEveryOffset(t *testing.T) {
	const n = 12
	h, seg := buildLog(t, n)

	whole, err := os.ReadFile(seg)
	if err != nil {
		t.Fatal(err)
	}
	ends := recordEnds(t, seg)
	if len(ends) != n {
		t.Fatalf("scanned %d records, want %d", len(ends), n)
	}

	// Every truncation point, including the ones that cut a record in half and
	// the ones that cut the header. A process killed mid-append leaves the file
	// at one of these, and none of them may fail the open.
	for cut := 0; cut <= len(whole); cut++ {
		t.Run(fmt.Sprintf("cut=%d", cut), func(t *testing.T) {
			dir := t.TempDir()
			base := filepath.Join(dir, "snap.mindb.wal")
			if err := os.WriteFile(segmentName(base, 1), whole[:cut], 0o644); err != nil {
				t.Fatal(err)
			}

			opts := Options{
				Dims:     walTestDims,
				Capacity: h.capacity,
				Snapshot: filepath.Join(dir, "snap.mindb"),
				WAL:      base,
			}
			e, rec, err := Open(opts)
			if err != nil {
				t.Fatalf("open after cutting at %d: %v", cut, err)
			}
			defer e.Close()

			var want int
			for _, end := range ends {
				if end <= int64(cut) {
					want++
				}
			}
			if rec.Records != want {
				t.Errorf("replayed %d records, want %d", rec.Records, want)
			}
			if e.Len() != want {
				t.Errorf("engine holds %d vectors, want %d", e.Len(), want)
			}
			for i := 0; i < want; i++ {
				if len(e.Get([]string{fmt.Sprintf("k%02d", i)})) != 1 {
					t.Fatalf("k%02d is missing, so the replayed records are not a prefix", i)
				}
			}
		})
	}
}

func TestBitFlipEndsTheLogThere(t *testing.T) {
	const n = 8
	h, seg := buildLog(t, n)
	ends := recordEnds(t, seg)

	// Flip a bit in the body of record 4. The checksum covers the length field
	// as well as the body, so this is caught whether the damage lands in either.
	whole, err := os.ReadFile(seg)
	if err != nil {
		t.Fatal(err)
	}
	victim := ends[3] + walRecordHeader + 2
	whole[victim] ^= 0x10
	if err := os.WriteFile(seg, whole, 0o644); err != nil {
		t.Fatal(err)
	}

	e, rec, err := Open(h.options())
	if err != nil {
		t.Fatalf("open over a damaged record: %v", err)
	}
	if !rec.Truncated {
		t.Error("a flipped bit was not reported as truncation")
	}
	if rec.Records != 4 {
		t.Errorf("replayed %d records, want 4", rec.Records)
	}
	if e.Len() != 4 {
		t.Errorf("engine holds %d vectors, want 4", e.Len())
	}
	want := contents(t, e)
	e.Close()

	// The decision has to stick. Recovery truncated the file, so opening the
	// same directory again must not replay the records the first open threw
	// away -- two recoveries of one set of files cannot disagree.
	again, rec, err := Open(h.options())
	if err != nil {
		t.Fatalf("second open: %v", err)
	}
	defer again.Close()
	if rec.Truncated {
		t.Error("the second open truncated again, so the first one did not finish the job")
	}
	sameContents(t, contents(t, again), want)
}

func TestSaveRotatesAndRetiresSegments(t *testing.T) {
	h := newWALHarness(t)

	e, _ := h.open()
	for i := 0; i < 5; i++ {
		mustInsert(t, e, fmt.Sprintf("k%02d", i), walVec(i), nil)
	}
	if err := e.Save(h.snapshot); err != nil {
		t.Fatalf("save: %v", err)
	}

	segs, err := walSegments(h.wal)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 1 {
		t.Fatalf("after a save the log has %d segments, want 1: %v", len(segs), segs)
	}
	if segs[0] != segmentName(h.wal, 2) {
		t.Errorf("live segment is %s, want %s", segs[0], segmentName(h.wal, 2))
	}
	if _, err := os.Stat(metaPath(h.snapshot)); err != nil {
		t.Errorf("no meta beside the snapshot: %v", err)
	}

	// Writes after the rotation live only in the new segment.
	for i := 5; i < 9; i++ {
		mustInsert(t, e, fmt.Sprintf("k%02d", i), walVec(i), nil)
	}
	want := contents(t, e)
	e.Close()

	reopened, rec, err := Open(h.options())
	if err != nil {
		t.Fatalf("open after rotation: %v", err)
	}
	defer reopened.Close()
	if rec.Records != 4 {
		t.Errorf("replayed %d records, want the 4 written after the snapshot", rec.Records)
	}
	sameContents(t, contents(t, reopened), want)
}

func TestConcurrentWritesAllSurvive(t *testing.T) {
	const writers, each = 8, 25
	h := newWALHarness(t)
	e, _ := h.open()

	var wg sync.WaitGroup
	errs := make(chan error, writers*each)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				n := w*each + i
				if err := e.Insert(fmt.Sprintf("k%03d", n), walVec(n), []byte(fmt.Sprintf("p%03d", n))); err != nil {
					errs <- err
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("insert: %v", err)
	}

	want := contents(t, e)
	if len(want) != writers*each {
		t.Fatalf("engine holds %d vectors, want %d", len(want), writers*each)
	}
	e.Close()

	reopened, _ := h.open()
	defer reopened.Close()
	sameContents(t, contents(t, reopened), want)
}

func TestLogOrderMatchesApplyOrder(t *testing.T) {
	const writers, each = 8, 40
	h := newWALHarness(t)
	e, _ := h.open()

	// Every writer fights over one id. Whichever write lands last in the slab
	// must also be the last one in the log, or the reopened engine disagrees
	// with the engine that acknowledged the writes.
	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				n := w*each + i
				if err := e.Insert("contested", walVec(n), []byte(fmt.Sprintf("w%d-%d", w, i))); err != nil {
					t.Errorf("insert: %v", err)
					return
				}
			}
		}(w)
	}
	wg.Wait()

	want := contents(t, e)
	e.Close()

	reopened, _ := h.open()
	defer reopened.Close()
	sameContents(t, contents(t, reopened), want)
}

func TestDimsChangeRefusesTheLog(t *testing.T) {
	h := newWALHarness(t)
	e, _ := h.open()
	mustInsert(t, e, "a", walVec(1), nil)
	e.Close()

	opts := h.options()
	opts.Dims = walTestDims * 2
	e, _, err := Open(opts)
	if err == nil {
		e.Close()
		t.Fatal("opened a log written at a different dimension")
	}
	if !errors.Is(err, ErrWALDimsChange) {
		t.Errorf("error is %v, want %v", err, ErrWALDimsChange)
	}
}

func TestOlderSnapshotWithNewerLogRefusesToStart(t *testing.T) {
	h := newWALHarness(t)

	e, _ := h.open()
	mustInsert(t, e, "a", walVec(1), nil)
	if err := e.Save(h.snapshot); err != nil {
		t.Fatalf("first save: %v", err)
	}

	// Keep the older snapshot and its meta, the way a restore from backup would.
	old := filepath.Join(h.dir, "old.mindb")
	copyFile(t, h.snapshot, old)
	copyFile(t, metaPath(h.snapshot), metaPath(old))

	mustInsert(t, e, "b", walVec(2), nil)
	if err := e.Save(h.snapshot); err != nil {
		t.Fatalf("second save: %v", err)
	}
	mustInsert(t, e, "c", walVec(3), nil)
	e.Close()

	// Restoring the older snapshot next to a log that continues the newer one.
	// Replaying it would apply records from a history this snapshot never had.
	copyFile(t, old, h.snapshot)
	copyFile(t, metaPath(old), metaPath(h.snapshot))

	e, _, err := Open(h.options())
	if err == nil {
		e.Close()
		t.Fatal("started on a snapshot the log does not continue")
	}
	if !errors.Is(err, ErrRunIDMismatch) {
		t.Fatalf("error is %v, want %v", err, ErrRunIDMismatch)
	}
	// The error has to say how to get out of this, since the operator is
	// holding two files that are each individually fine.
	if msg := err.Error(); !strings.Contains(msg, h.wal) {
		t.Errorf("error does not name the log to delete: %s", msg)
	}

	// And the fix it names works: drop the log, keep the snapshot.
	segs, err := walSegments(h.wal)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range segs {
		os.Remove(s)
	}
	restored, _, err := Open(h.options())
	if err != nil {
		t.Fatalf("open with the log removed: %v", err)
	}
	defer restored.Close()
	if restored.Len() != 1 {
		t.Errorf("restored snapshot holds %d vectors, want 1", restored.Len())
	}
}

func TestLegacySnapshotWithoutMetaIsReplayedAndReported(t *testing.T) {
	h := newWALHarness(t)

	e, _ := h.open()
	mustInsert(t, e, "a", walVec(1), nil)
	if err := e.Save(h.snapshot); err != nil {
		t.Fatal(err)
	}
	mustInsert(t, e, "b", walVec(2), nil)
	want := contents(t, e)
	e.Close()

	// A snapshot written before meta files existed has no lineage to check.
	if err := os.Remove(metaPath(h.snapshot)); err != nil {
		t.Fatal(err)
	}

	reopened, rec, err := Open(h.options())
	if err != nil {
		t.Fatalf("open beside a snapshot with no meta: %v", err)
	}
	defer reopened.Close()
	if !rec.LegacyMeta {
		t.Error("a missing meta file was not reported")
	}
	sameContents(t, contents(t, reopened), want)
}

func TestWALRequiresASnapshotPath(t *testing.T) {
	dir := t.TempDir()
	e, _, err := Open(Options{Dims: walTestDims, Capacity: 16, WAL: filepath.Join(dir, "x.wal")})
	if err == nil {
		e.Close()
		t.Fatal("a log with nothing to retire it was accepted")
	}
}

func TestWriteFailureMarksTheLogUnhealthy(t *testing.T) {
	h := newWALHarness(t)
	e, _ := h.open()
	defer e.Close()

	mustInsert(t, e, "a", walVec(1), nil)
	if !e.WALHealthy() {
		t.Fatal("healthy is false before anything has failed")
	}

	// Pull the file out from under the writer, which is the closest thing to a
	// disk that has stopped accepting writes that a test can arrange portably.
	if err := e.wal.f.Close(); err != nil {
		t.Fatal(err)
	}

	if err := e.Insert("b", walVec(2), nil); err == nil {
		t.Fatal("insert was acknowledged after the log stopped accepting writes")
	}
	if e.WALHealthy() {
		t.Error("healthy is still true after a failed write")
	}
	if st := e.Stats(); st.WALHealthy || !st.WALEnabled {
		t.Errorf("Stats reports enabled=%t healthy=%t, want true/false", st.WALEnabled, st.WALHealthy)
	}
	// The callback runs on its own goroutine, so this waits for it rather than
	// polling: a readiness probe that never fires is the failure being tested.
	select {
	case <-h.faults:
	case <-time.After(5 * time.Second):
		t.Error("OnWALFault was not called")
	}

	// Sticky: a later write must not be acknowledged on the strength of an
	// fsync that cannot speak for the record that was already lost.
	if err := e.Insert("c", walVec(3), nil); err == nil {
		t.Error("a later insert was acknowledged while the log is unhealthy")
	}
}

func copyFile(t *testing.T, from, to string) {
	t.Helper()
	b, err := os.ReadFile(from)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(to, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// --- killing the process for real ---------------------------------------

// The tests above close the engine or damage the file by hand. These ones run
// a child process that writes until it is killed outright, because the thing
// being tested is what an unclean stop leaves behind, and a clean Close is the
// one case that cannot demonstrate it.

const (
	crashDirEnv   = "MINDB_WAL_CRASH_DIR"
	crashCountEnv = "MINDB_WAL_CRASH_AFTER"
)

func TestMain(m *testing.M) {
	if dir := os.Getenv(crashDirEnv); dir != "" {
		crashChild(dir, os.Getenv(crashCountEnv))
		return
	}
	os.Exit(m.Run())
}

// crashChild inserts until it is killed, or exits hard after n writes. Either
// way it never closes the engine, so nothing is flushed on the way out.
func crashChild(dir, after string) {
	e, _, err := Open(Options{
		Dims:     walTestDims,
		Capacity: 4096,
		Snapshot: filepath.Join(dir, "snap.mindb"),
		WAL:      filepath.Join(dir, "snap.mindb.wal"),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "child open: %v\n", err)
		os.Exit(2)
	}

	var stop int
	fmt.Sscanf(after, "%d", &stop)

	for i := 0; ; i++ {
		id := fmt.Sprintf("k%04d", i)
		if err := e.Insert(id, walVec(i), []byte(id)); err != nil {
			fmt.Fprintf(os.Stderr, "child insert: %v\n", err)
			os.Exit(2)
		}
		// Printed only after the insert returned, so every id the parent reads
		// is one the engine acknowledged.
		fmt.Fprintln(os.Stdout, id)

		if stop > 0 && i+1 >= stop {
			os.Exit(0) // no Close, no final flush
		}
	}
}

func TestHardExitKeepsEveryAcknowledgedWrite(t *testing.T) {
	dir := t.TempDir()
	const n = 40

	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(),
		crashDirEnv+"="+dir,
		fmt.Sprintf("%s=%d", crashCountEnv, n))
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("child: %v", err)
	}
	if got := strings.Count(string(out), "\n"); got != n {
		t.Fatalf("child acknowledged %d writes, want %d", got, n)
	}

	e, rec, err := Open(Options{
		Dims:     walTestDims,
		Capacity: 4096,
		Snapshot: filepath.Join(dir, "snap.mindb"),
		WAL:      filepath.Join(dir, "snap.mindb.wal"),
	})
	if err != nil {
		t.Fatalf("open after a hard exit: %v", err)
	}
	defer e.Close()

	if rec.Records != n {
		t.Errorf("replayed %d records, want %d", rec.Records, n)
	}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("k%04d", i)
		got := e.Get([]string{id})
		if len(got) != 1 {
			t.Fatalf("%s did not survive the exit", id)
		}
		if string(got[0].Payload) != id {
			t.Errorf("%s payload = %q", id, got[0].Payload)
		}
	}
}

func TestKilledMidWriteKeepsEveryAcknowledgedWrite(t *testing.T) {
	dir := t.TempDir()
	const want = 25

	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), crashDirEnv+"="+dir)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	// Read acknowledged ids until there are enough to be interesting, then
	// kill the process in the middle of whatever it is doing.
	acked := make([]string, 0, want)
	scan := bufio.NewScanner(stdout)
	for scan.Scan() && len(acked) < want {
		acked = append(acked, scan.Text())
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Wait()

	if len(acked) != want {
		t.Fatalf("read %d acknowledged ids before the kill, want %d", len(acked), want)
	}

	e, _, err := Open(Options{
		Dims:     walTestDims,
		Capacity: 4096,
		Snapshot: filepath.Join(dir, "snap.mindb"),
		WAL:      filepath.Join(dir, "snap.mindb.wal"),
	})
	if err != nil {
		t.Fatalf("open after a kill: %v", err)
	}
	defer e.Close()

	// Writes still in flight when the process died may or may not be here, and
	// the test says nothing about them. The acknowledged ones are not optional.
	for _, id := range acked {
		if len(e.Get([]string{id})) != 1 {
			t.Fatalf("%s was acknowledged and then lost", id)
		}
	}
}

func countLines(s string) int {
	var n int
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			n++
		}
	}
	return n
}
