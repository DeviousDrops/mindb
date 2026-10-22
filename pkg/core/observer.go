package core

import (
	"errors"
	"os"
	"time"
)

// Observer receives measurements from the engine's write, search and snapshot
// paths. Set it with Options.Observer; nil, the default, costs nothing — not
// even the clock reads.
//
// It is an interface rather than a metrics type so that pkg/core does not
// depend on any metrics library. pkg/telemetry is the implementation that turns
// these events into Prometheus metrics.
//
// Methods are called synchronously on the path being measured, outside the
// engine's locks. They must be cheap and must not call back into the engine.
type Observer interface {
	WALSync(WALSync)
	Search(SearchEvent)
	Snapshot(SnapshotEvent)

	// WriteApply is the time from an Insert or Delete asking for the write
	// lock to its change being in the slab, before it is logged. Nearly all of
	// it is waiting: behind other writers for writeMu, then behind running
	// searches for the slab lock. When this dominates Insert latency, writers
	// are reaching the log one at a time and group commit cannot batch them.
	WriteApply(time.Duration)
}

// WALSync describes one flush-and-fsync of the write-ahead log.
//
// Under group commit, one sync makes a whole batch of writers durable, so
// Records is the batch size: the number this is meant to show going up under
// concurrent load while the number of fsyncs stays flat.
type WALSync struct {
	Records int   // records this sync made durable
	Bytes   int64 // bytes those records occupy in the log

	// Write is the time to copy the buffered records into the page cache;
	// Fsync is the time for the device to make them durable. They are split
	// because they fail in different ways: a slow Write is memory pressure, a
	// slow Fsync is the disk.
	Write time.Duration
	Fsync time.Duration

	// Rotation is true for the sync that closes a segment before a snapshot,
	// rather than one a writer is waiting on.
	Rotation bool

	Err error
}

// SearchPath is which algorithm served a query.
type SearchPath uint8

const (
	// PathScan is the plain float32 scan over every live vector.
	PathScan SearchPath = iota
	// PathCascade is bound-and-refine: an int8 pass that prunes, then an exact
	// float32 rescore of the survivors.
	PathCascade
	// PathFallback is the cascade abandoning its shortlist because too many
	// vectors survived pruning, and running a plain scan instead. Results are
	// identical; it only costs time, which is why it is worth seeing.
	PathFallback
)

func (p SearchPath) String() string {
	switch p {
	case PathScan:
		return "scan"
	case PathCascade:
		return "cascade"
	case PathFallback:
		return "fallback"
	}
	return "unknown"
}

// SearchEvent describes one Search call.
type SearchEvent struct {
	Path SearchPath

	// Duration covers the whole call, including the wait for the read lock,
	// because that is what the caller experiences.
	Duration time.Duration

	// Live is the number of vectors the query ran against.
	Live int

	// Survivors is how many vectors passed the cascade's pruning bound and
	// were rescored exactly. Zero on the scan path. Survivors/Live is the
	// pruning rate, which decides whether the cascade is paying for itself.
	Survivors int
}

// SnapshotEvent describes one Save.
type SnapshotEvent struct {
	Duration time.Duration // the whole Save: rotate, write, fsync, rename, meta
	Bytes    int64         // size of the snapshot file written
	Err      error
}

// WALDiskUsage reports how many log segments exist and their total size.
//
// It reads the directory, not engine state, and takes no lock. A segment that
// a concurrent Save deletes between listing and stat is skipped rather than
// reported as an error: it is gone, which is the answer.
func (e *Engine) WALDiskUsage() (segments int, bytes int64, err error) {
	if e.wal == nil {
		return 0, 0, nil
	}
	paths, err := walSegments(e.wal.base)
	if err != nil {
		return 0, 0, err
	}
	for _, p := range paths {
		info, err := os.Stat(p)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return 0, 0, err
		}
		segments++
		bytes += info.Size()
	}
	return segments, bytes, nil
}

// WALBase is the base path log segments are named after, or "" when the log is
// off. The server uses it to find which block device the log lives on.
func (e *Engine) WALBase() string {
	if e.wal == nil {
		return ""
	}
	return e.wal.base
}
