package core

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// A run id ties a snapshot to the write-ahead log segments that continue it.
//
// Every completed Save mints a new one, stamps it into the segment it rotates
// to, and records it in a companion `<snapshot>.meta` file. At startup a log is
// accepted only if some retained segment carries the meta's run id, which is
// exactly the condition "the log reaches back to at least this snapshot".
//
// The case this catches is restoring an older snapshot by hand and leaving the
// live log in place. The segment covering the gap between them was deleted when
// the newer snapshot completed, so replay would produce a state that never
// existed. Nothing about the two files looks wrong without the id.
//
// It lives beside the snapshot rather than inside it so that the snapshot format
// stays at v1 and every file already in the field still loads.
const runIDLen = 16

type runID [runIDLen]byte

var ErrRunIDMismatch = errors.New("mindb: write-ahead log does not belong to this snapshot")

func newRunID() (runID, error) {
	var r runID
	if _, err := rand.Read(r[:]); err != nil {
		return r, fmt.Errorf("mindb: generate run id: %w", err)
	}
	return r, nil
}

func (r runID) String() string { return hex.EncodeToString(r[:]) }

func parseRunID(s string) (runID, error) {
	var r runID
	b, err := hex.DecodeString(s)
	if err != nil {
		return r, fmt.Errorf("mindb: run id is not hex: %w", err)
	}
	if len(b) != runIDLen {
		return r, fmt.Errorf("mindb: run id is %d bytes, want %d", len(b), runIDLen)
	}
	copy(r[:], b)
	return r, nil
}

// metaPath returns the companion file for a snapshot path.
func metaPath(snapshot string) string { return snapshot + ".meta" }

const metaHeader = "mindb-meta v1"

// writeMeta replaces the meta file atomically, on the same tmp/rename/fsync
// sequence Save uses. A half-written meta would be worse than none: a missing
// one is treated as legacy, a corrupt one refuses to start.
func writeMeta(path string, run runID) (err error) {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp*")
	if err != nil {
		return fmt.Errorf("mindb: create temp meta: %w", err)
	}
	tmpName := tmp.Name()
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmpName)
		}
	}()

	body := metaHeader + "\nrun_id " + run.String() + "\n"
	if _, err = tmp.WriteString(body); err != nil {
		return fmt.Errorf("mindb: write meta: %w", err)
	}
	if err = tmp.Sync(); err != nil {
		return fmt.Errorf("mindb: fsync meta: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("mindb: close meta: %w", err)
	}
	if err = os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("mindb: rename meta into place: %w", err)
	}
	return syncDir(dir)
}

// readMeta returns the run id recorded beside a snapshot.
//
// A missing file returns os.ErrNotExist, which the caller treats as a snapshot
// written before meta files existed. Anything else is an error: a meta that is
// present but unreadable means something is wrong that guessing will not fix.
func readMeta(path string) (runID, error) {
	var run runID

	f, err := os.Open(path)
	if err != nil {
		return run, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	if !sc.Scan() || strings.TrimSpace(sc.Text()) != metaHeader {
		return run, fmt.Errorf("mindb: %s is not a MinDB meta file", path)
	}
	for sc.Scan() {
		key, value, ok := strings.Cut(strings.TrimSpace(sc.Text()), " ")
		if !ok || key != "run_id" {
			continue
		}
		return parseRunID(value)
	}
	if err := sc.Err(); err != nil {
		return run, fmt.Errorf("mindb: read %s: %w", path, err)
	}
	return run, fmt.Errorf("mindb: %s has no run_id", path)
}
