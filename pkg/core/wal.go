package core

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
)

// wal is an append-only log of the operations that mutate the slab.
//
// # This is a redo log, not a write-ahead log in the strict sense
//
// The engine applies an operation to the slab first and logs it second
// (pkg/core/engine.go, Insert and Delete). The name is kept because that is what
// everyone calls this file, but the ordering is Redis's AOF, not ARIES.
//
// The guarantee that matters is unchanged: an operation is acknowledged only
// after its record is durable, so **every acked write survives a crash**. What
// differs is that an *unacked* write becomes visible to readers before it is
// durable, so a Search running concurrently with an in-flight Insert can return
// a vector that a crash then erases. No client was ever told that Insert
// succeeded; from any client's point of view it was in flight, and an in-flight
// write may or may not be observed and may or may not survive.
//
// True log-before-apply would need the slot reserved before the append, because
// store's only possible failure is ErrCapacityExceeded and by then the record is
// already durable. That means splitting the allocator into reserve and fill, and
// it means a batch of group-committed writers has to be applied in log order by
// one of them rather than each applying its own. See DECISIONS.md.
//
// # Segments
//
// The log is a sequence of numbered segments. Save rotates to a new one before
// writing the snapshot and deletes the old one after, which keeps the invariant
// replay depends on: the retained segments always cover a contiguous range of
// history ending at now and beginning at or before the snapshot's point.
type wal struct {
	dims int
	base string

	mu   sync.Mutex
	cond *sync.Cond

	f   *os.File
	buf *bufio.Writer
	seg uint64
	run runID

	// nextSeq counts buffered records; syncedSeq counts durable ones. A writer
	// is done when syncedSeq reaches its own sequence number, whoever got it
	// there.
	nextSeq   uint64
	syncedSeq uint64
	syncing   bool
	syncErr   error

	// healthy goes false on the first write or fsync failure and never goes back
	// on. A retry that succeeds proves nothing: on Linux a failed fsync may have
	// already dropped the dirty pages it could not write, so the data is gone and
	// the next call has nothing left to fail on. For the same reason syncErr is
	// sticky once this is false: every write from then on is refused rather than
	// acknowledged on the strength of an fsync that cannot speak for the ones
	// before it. Reads carry on; the process does not halt.
	healthy atomic.Bool
	onFault func()
	faulted atomic.Bool

	// syncAlone stops batches from forming, so the fsync count tracks the write
	// count. Only BenchmarkWALInsert sets it, to measure what group commit is
	// worth against the thing it replaced.
	syncAlone bool
}

// walBufferSize is one segment's write buffer. Group commit means a batch is
// flushed as one write, so this wants to be comfortably larger than a record.
const walBufferSize = 1 << 20

// segmentName is the path of segment n.
func segmentName(base string, n uint64) string {
	return fmt.Sprintf("%s.%06d", base, n)
}

// walSegments lists existing segments in replay order.
func walSegments(base string) ([]string, error) {
	matches, err := filepath.Glob(base + ".??????")
	if err != nil {
		return nil, fmt.Errorf("mindb: list write-ahead log segments: %w", err)
	}
	sort.Strings(matches) // zero-padded, so lexical order is numeric order
	return matches, nil
}

// lastSegmentNumber returns the highest segment number present, or 0 if none.
func lastSegmentNumber(base string) (uint64, error) {
	segs, err := walSegments(base)
	if err != nil {
		return 0, err
	}
	if len(segs) == 0 {
		return 0, nil
	}
	var n uint64
	_, err = fmt.Sscanf(filepath.Ext(segs[len(segs)-1]), ".%d", &n)
	if err != nil {
		return 0, fmt.Errorf("mindb: unparseable segment name %q", segs[len(segs)-1])
	}
	return n, nil
}

// openWAL opens segment n for append, creating it with a header if it is new.
//
// An existing segment keeps its own run id, which is the one replay validated;
// run is used only when the file has to be created.
func openWAL(base string, n uint64, dims int, run runID, onFault func()) (*wal, error) {
	w := &wal{dims: dims, base: base, seg: n, run: run, onFault: onFault}
	w.cond = sync.NewCond(&w.mu)
	w.healthy.Store(true)

	if err := w.openSegment(n, run); err != nil {
		return nil, err
	}
	return w, nil
}

// openSegment attaches the writer to segment n. Caller must not hold w.mu.
func (w *wal) openSegment(n uint64, run runID) error {
	path := segmentName(w.base, n)

	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return fmt.Errorf("mindb: open write-ahead log %s: %w", path, err)
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return fmt.Errorf("mindb: stat write-ahead log %s: %w", path, err)
	}

	switch {
	case info.Size() == 0:
		hdr := appendSegmentHeader(nil, w.dims, run)
		if _, err := f.Write(hdr); err != nil {
			f.Close()
			return fmt.Errorf("mindb: write write-ahead log header: %w", err)
		}
		if err := f.Sync(); err != nil {
			f.Close()
			return fmt.Errorf("mindb: fsync write-ahead log header: %w", err)
		}
		if err := syncDir(filepath.Dir(path)); err != nil {
			f.Close()
			return err
		}
	default:
		hdr := make([]byte, walHeaderSize)
		if _, err := io.ReadFull(f, hdr); err != nil {
			f.Close()
			return fmt.Errorf("mindb: read write-ahead log header %s: %w", path, err)
		}
		if run, err = parseSegmentHeader(hdr, w.dims); err != nil {
			f.Close()
			return err
		}
		if _, err := f.Seek(0, io.SeekEnd); err != nil {
			f.Close()
			return fmt.Errorf("mindb: seek write-ahead log %s: %w", path, err)
		}
	}

	w.f = f
	w.buf = bufio.NewWriterSize(f, walBufferSize)
	w.seg = n
	w.run = run
	return nil
}

// buffer appends an encoded record and returns the sequence number that must be
// durable before it is safe to acknowledge.
//
// Callers must hold the engine's write mutex, so that the order records land in
// the log is the order they were applied to the slab.
func (w *wal) buffer(rec []byte) uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.nextSeq++
	if _, err := w.buf.Write(rec); err != nil {
		w.fault()
		w.syncErr = fmt.Errorf("mindb: append to write-ahead log: %w", err)
	}
	return w.nextSeq
}

// syncTo blocks until seq is durable, doing the fsync itself if nobody else is.
//
// The writer that arrives while no sync is in flight becomes the leader: it
// flushes everything buffered so far, records how far that reaches, and syncs
// with the lock released so later writers keep appending. They land in the next
// batch. Nothing here is timer-driven — a lone writer syncs immediately, and
// under load the batch is however many writers arrived during the last fsync.
func (w *wal) syncTo(seq uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	for w.syncedSeq < seq {
		if w.syncing {
			w.cond.Wait()
			continue
		}

		w.syncing = true
		target := w.nextSeq
		err := w.flushLocked()

		if err == nil {
			if w.syncAlone {
				err = w.f.Sync()
			} else {
				w.mu.Unlock()
				err = w.f.Sync()
				w.mu.Lock()
			}
		}

		w.syncing = false
		w.syncedSeq = target
		switch {
		case err != nil:
			w.fault()
			w.syncErr = fmt.Errorf("mindb: fsync write-ahead log: %w", err)
		case w.healthy.Load():
			w.syncErr = nil
		}
		w.cond.Broadcast()
	}
	return w.syncErr
}

// flushLocked empties the write buffer into the file. Caller must hold w.mu.
func (w *wal) flushLocked() error {
	if err := w.buf.Flush(); err != nil {
		return err
	}
	return nil
}

// rotate makes everything written so far durable, then starts segment n+1 under
// a new run id. The retired segment's path is returned so the caller can delete
// it once the snapshot that supersedes it is on disk.
//
// Callers must hold the engine's write mutex, so the rotation lands at a
// definite point in the write order rather than in the middle of a batch.
func (w *wal) rotate(run runID) (retired string, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err = w.flushLocked(); err != nil {
		w.fault()
		return "", fmt.Errorf("mindb: flush write-ahead log before rotate: %w", err)
	}
	if err = w.f.Sync(); err != nil {
		w.fault()
		return "", fmt.Errorf("mindb: fsync write-ahead log before rotate: %w", err)
	}
	if err = w.f.Close(); err != nil {
		w.fault()
		return "", fmt.Errorf("mindb: close write-ahead log: %w", err)
	}

	// Everything buffered is now on disk, so anybody still waiting on a sync is
	// satisfied by the rotation itself.
	retired = segmentName(w.base, w.seg)
	w.syncedSeq = w.nextSeq
	w.syncErr = nil
	w.cond.Broadcast()

	if err = w.openSegment(w.seg+1, run); err != nil {
		w.fault()
		return "", err
	}
	return retired, nil
}

// close flushes and closes the current segment.
func (w *wal) close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.f == nil {
		return nil
	}
	err := w.flushLocked()
	if syncErr := w.f.Sync(); err == nil {
		err = syncErr
	}
	if closeErr := w.f.Close(); err == nil {
		err = closeErr
	}
	w.f = nil
	return err
}

// fault marks the log unhealthy and notifies the server once.
//
// Never reset: see the healthy field. The readiness probe reads this, so the
// pod stops taking traffic rather than quietly serving writes it cannot persist.
func (w *wal) fault() {
	w.healthy.Store(false)
	if w.onFault != nil && w.faulted.CompareAndSwap(false, true) {
		go w.onFault()
	}
}

// currentRun reports the run id stamped in the segment now being written.
func (w *wal) currentRun() runID {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.run
}
