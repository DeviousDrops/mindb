package core

import (
	"errors"
	"fmt"
	"os"
)

// Options configures Open.
type Options struct {
	Dims     int
	Capacity int

	// Snapshot is the snapshot file path. Empty disables persistence entirely.
	Snapshot string

	// WAL is the base path for write-ahead log segments, which are named
	// <WAL>.000001 and up. Empty disables logging. It requires Snapshot: without
	// snapshots nothing ever retires a segment and the log grows forever.
	WAL string

	// OnWALFault is called once, from its own goroutine, the first time the log
	// cannot be written or synced. The server uses it to fail its readiness
	// probe. Never called again, because the log never becomes healthy again.
	OnWALFault func()
}

// Open builds an engine from whatever durable state exists: a snapshot, a log
// continuing it, both, or neither.
//
// The order is load, then replay, then attach. Replay goes through the engine's
// internal write path rather than Insert and Delete, so the log being replayed
// is not written back to itself.
func Open(opts Options) (*Engine, WALRecovery, error) {
	var rec WALRecovery

	if opts.WAL != "" && opts.Snapshot == "" {
		return nil, rec, errors.New("mindb: a write-ahead log requires a snapshot path to retire it against")
	}

	e, hadSnapshot, err := loadOrCreate(opts)
	if err != nil {
		return nil, rec, err
	}
	if opts.WAL == "" {
		return e, rec, nil
	}

	// The run id recorded beside the snapshot. Absent when this engine is new,
	// and absent for snapshots written before meta files existed -- the latter
	// is worth saying out loud, because it is the one case where the log is
	// replayed without being checked against the snapshot it continues.
	var (
		expect   runID
		checkRun bool
	)
	if hadSnapshot {
		switch expect, err = readMeta(metaPath(opts.Snapshot)); {
		case err == nil:
			checkRun = true
		case errors.Is(err, os.ErrNotExist):
			rec.LegacyMeta = true
		default:
			return nil, rec, err
		}
	}

	rec, err = replayWAL(e, opts.WAL, expect, checkRun)
	if err != nil {
		return nil, rec, err
	}
	rec.LegacyMeta = rec.LegacyMeta || (hadSnapshot && !checkRun)

	if err := e.attachWAL(opts, expect, checkRun); err != nil {
		return nil, rec, err
	}
	return e, rec, nil
}

// loadOrCreate returns an engine from the snapshot at opts.Snapshot, or an empty
// one, reporting which happened.
func loadOrCreate(opts Options) (*Engine, bool, error) {
	if opts.Snapshot == "" {
		e, err := New(opts.Dims, opts.Capacity)
		return e, false, err
	}

	e, err := Load(opts.Snapshot, opts.Capacity)
	switch {
	case err == nil:
		return e, true, nil
	case errors.Is(err, os.ErrNotExist):
		e, err := New(opts.Dims, opts.Capacity)
		return e, false, err
	default:
		// Refuse to start rather than silently discarding a corrupt snapshot:
		// booting empty would look like data loss and be indistinguishable from
		// a first run.
		return nil, false, fmt.Errorf("mindb: load snapshot %s: %w", opts.Snapshot, err)
	}
}

// attachWAL opens the segment new records will be appended to.
//
// An existing segment keeps the run id in its own header. A brand new one takes
// the snapshot's run id when there is one, so that the "some retained segment
// carries the meta's run id" check still holds on the next restart -- a fresh
// id here would make the engine refuse to start on files it just wrote itself.
func (e *Engine) attachWAL(opts Options, expect runID, haveExpect bool) error {
	n, err := lastSegmentNumber(opts.WAL)
	if err != nil {
		return err
	}

	run := expect
	if n == 0 {
		n = 1
		if !haveExpect {
			if run, err = newRunID(); err != nil {
				return err
			}
		}
	}

	w, err := openWAL(opts.WAL, n, e.dims, run, opts.OnWALFault)
	if err != nil {
		return err
	}
	e.wal = w
	return nil
}

// Close flushes and closes the write-ahead log. The engine is unusable after.
//
// Callers that want the log retired rather than merely flushed should Save
// first; Close only makes what is already there durable.
func (e *Engine) Close() error {
	if e.wal == nil {
		return nil
	}
	return e.wal.close()
}

// WALEnabled reports whether writes are being logged.
func (e *Engine) WALEnabled() bool { return e.wal != nil }

// WALHealthy reports whether every record so far reached the disk.
//
// False is sticky. A write or fsync that fails and then succeeds on retry proves
// nothing: on Linux a failed fsync may already have dropped the dirty pages it
// could not write, so the data is gone and the next call has nothing left to
// fail on. Once this is false the process should stop taking traffic.
func (e *Engine) WALHealthy() bool {
	if e.wal == nil {
		return true
	}
	return e.wal.healthy.Load()
}
