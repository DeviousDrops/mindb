package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
)

// WALRecovery is what replay found, for the startup log.
//
// Truncated is the one field worth alerting on: it means the log ended in a
// damaged record and everything from there on was discarded.
type WALRecovery struct {
	Segments  int
	Records   int
	Truncated bool

	// LegacyMeta is set when a snapshot exists with no companion meta file, so
	// the log could not be checked against it. Snapshots written before meta
	// files existed land here.
	LegacyMeta bool
}

// replayWAL applies every segment under base to e, in order.
//
// expect is the run id recorded beside the snapshot; when checkRun is false
// there was no meta file to check against. The log is accepted only if some
// retained segment carries expect, which is the condition "the log reaches back
// to at least this snapshot" — see meta.go.
func replayWAL(e *Engine, base string, expect runID, checkRun bool) (WALRecovery, error) {
	var rec WALRecovery

	segments, err := walSegments(base)
	if err != nil {
		return rec, err
	}
	if segments, err = dropStubSegments(segments, &rec); err != nil {
		return rec, err
	}
	if len(segments) == 0 {
		return rec, nil
	}

	runs := make([]runID, len(segments))
	for i, path := range segments {
		if runs[i], err = readSegmentRun(path, e.dims); err != nil {
			return rec, err
		}
	}

	if checkRun && !containsRun(runs, expect) {
		return rec, fmt.Errorf(
			"%w: snapshot expects run %s, log carries %s. "+
				"Restoring a snapshot means replacing the snapshot, its .meta and the log together, "+
				"or deleting %s.?????? so the snapshot stands alone",
			ErrRunIDMismatch, expect, runs[len(runs)-1], base)
	}

	for i, path := range segments {
		n, good, damaged, err := replaySegment(e, path)
		rec.Segments++
		rec.Records += n
		if err != nil {
			return rec, err
		}
		if !damaged {
			continue
		}

		// Stop at the first damaged record and make that decision stick: the
		// tail of this segment goes, and so do the segments after it. Leaving
		// them would mean the next restart replays records this one discarded,
		// so two recoveries of the same files would disagree.
		rec.Truncated = true
		if err := os.Truncate(path, good); err != nil {
			return rec, fmt.Errorf("mindb: truncate damaged write-ahead log %s: %w", path, err)
		}
		for _, orphan := range segments[i+1:] {
			if err := os.Remove(orphan); err != nil {
				return rec, fmt.Errorf("mindb: remove unreachable write-ahead log %s: %w", orphan, err)
			}
		}
		return rec, nil
	}
	return rec, nil
}

// dropStubSegments removes segments too short to hold a header, along with
// everything after them.
//
// A segment is created and its header synced in one step, so a file shorter
// than a header is a crash caught between those two -- no record can have been
// written to it, and nothing acknowledged is lost by removing it. What can be
// lost is a later segment, which is why that case, and only that case, is
// reported as truncation.
func dropStubSegments(segments []string, rec *WALRecovery) ([]string, error) {
	for i, path := range segments {
		info, err := os.Stat(path)
		if err != nil {
			return nil, fmt.Errorf("mindb: stat write-ahead log %s: %w", path, err)
		}
		if info.Size() >= walHeaderSize {
			continue
		}
		if len(segments) > i+1 {
			rec.Truncated = true
		}
		for _, orphan := range segments[i:] {
			if err := os.Remove(orphan); err != nil {
				return nil, fmt.Errorf("mindb: remove headerless write-ahead log %s: %w", orphan, err)
			}
		}
		return segments[:i], nil
	}
	return segments, nil
}

// readSegmentRun validates a segment header and returns its run id.
func readSegmentRun(path string, dims int) (runID, error) {
	var run runID

	f, err := os.Open(path)
	if err != nil {
		return run, fmt.Errorf("mindb: open write-ahead log %s: %w", path, err)
	}
	defer f.Close()

	hdr := make([]byte, walHeaderSize)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return run, fmt.Errorf("mindb: read write-ahead log header %s: %w", path, err)
	}
	run, err = parseSegmentHeader(hdr, dims)
	if err != nil {
		return run, fmt.Errorf("%s: %w", path, err)
	}
	return run, nil
}

func containsRun(runs []runID, want runID) bool {
	for _, r := range runs {
		if r == want {
			return true
		}
	}
	return false
}

// replaySegment applies one segment, returning how many records were applied,
// the offset just past the last good one, and whether it ended in damage.
//
// A damaged record is not an error. Killing a process mid-append leaves a
// partial record at the end of the file, which is the expected way for a log to
// end, not a sign of corruption. An error here means the filesystem failed or a
// record could not be applied.
func replaySegment(e *Engine, path string) (applied int, good int64, damaged bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, false, fmt.Errorf("mindb: open write-ahead log %s: %w", path, err)
	}
	defer f.Close()

	r := newWALScanner(f, e.dims)
	if err := r.skipHeader(); err != nil {
		return 0, 0, false, fmt.Errorf("mindb: %s: %w", path, err)
	}

	for {
		rec, ok := r.next()
		if !ok {
			return applied, r.good, r.damaged, r.err
		}
		switch rec.op {
		case opInsert:
			// store, not Insert: the log holds normalized vectors and it adopts
			// the slice, which the scanner allocated fresh for this record.
			if err := e.store(rec.id, rec.vector, rec.payload); err != nil {
				return applied, r.good, false, fmt.Errorf("mindb: replay %s record %d: %w", path, applied+1, err)
			}
		case opDelete:
			// No log is attached during replay, so this cannot fail.
			_, _ = e.Delete(rec.id)
		}
		applied++
		r.good = r.off
	}
}

// walScanner reads framed records, stopping at the first one it cannot trust.
type walScanner struct {
	f    *os.File
	dims int

	off     int64 // bytes consumed
	good    int64 // offset just past the last record the caller accepted
	damaged bool
	err     error

	hdr []byte
}

func newWALScanner(f *os.File, dims int) *walScanner {
	return &walScanner{f: f, dims: dims, hdr: make([]byte, walRecordHeader)}
}

func (s *walScanner) skipHeader() error {
	hdr := make([]byte, walHeaderSize)
	if _, err := io.ReadFull(s.f, hdr); err != nil {
		return fmt.Errorf("read header: %w", err)
	}
	if _, err := parseSegmentHeader(hdr, s.dims); err != nil {
		return err
	}
	s.off = walHeaderSize
	s.good = walHeaderSize
	return nil
}

// next returns the next record, or false at the end of the log — clean or torn.
func (s *walScanner) next() (walRecord, bool) {
	var rec walRecord

	switch _, err := io.ReadFull(s.f, s.hdr); {
	case errors.Is(err, io.EOF):
		return rec, false // clean end: nothing at all where a record would start
	case err != nil:
		s.damaged = true // a partial header is a torn tail
		return rec, false
	}

	length := binary.LittleEndian.Uint32(s.hdr[:4])
	want := binary.LittleEndian.Uint32(s.hdr[4:])
	if length > walMaxRecord {
		s.damaged = true
		return rec, false
	}

	body := make([]byte, length)
	if _, err := io.ReadFull(s.f, body); err != nil {
		s.damaged = true
		return rec, false
	}

	crc := crc32.NewIEEE()
	crc.Write(s.hdr[:4])
	crc.Write(body)
	if crc.Sum32() != want {
		s.damaged = true
		return rec, false
	}

	rec, err := decodeBody(body, s.dims)
	if err != nil {
		// The checksum passed but the body is not a record this build can read.
		// Treating it as damage rather than an error keeps a forward-compatible
		// log from bringing the server down, at the cost of silently stopping.
		s.damaged = true
		return rec, false
	}

	s.off += int64(walRecordHeader) + int64(length)
	return rec, true
}
