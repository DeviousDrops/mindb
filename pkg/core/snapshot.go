package core

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	stdmath "math"
	"os"
	"path/filepath"
	"runtime"
)

// Snapshot file format. All integers little-endian.
//
//	magic     8 bytes  "MINDBSNP"
//	version   uint32
//	dims      uint32
//	count     uint32
//	records   count * { idLen uint32, id, payloadLen uint32, payload, dims*float32 }
//	crc32     uint32   IEEE checksum of every byte above
//
// Only float32 vectors are stored. The int8 codes the cascade uses are
// recomputed at load: persisting them would grow the file ~25% and re-quantizing
// costs ~300 ms at 100k x 768, which is noise next to reading 293 MB off disk.
// More usefully it decouples the on-disk format from the quantization scheme, so
// changing how codes are built does not invalidate every snapshot in the field.
const (
	snapshotVersion = 1
	headerSize      = 8 + 4 + 4 + 4
)

var snapshotMagic = [8]byte{'M', 'I', 'N', 'D', 'B', 'S', 'N', 'P'}

var (
	ErrBadMagic     = errors.New("mindb: not a MinDB snapshot")
	ErrBadVersion   = errors.New("mindb: unsupported snapshot version")
	ErrBadChecksum  = errors.New("mindb: snapshot checksum mismatch (file is corrupt or truncated)")
	ErrSnapshotSize = errors.New("mindb: snapshot does not fit in engine capacity")
)

// Save writes the engine to path atomically, and retires the write-ahead log
// records the snapshot now covers.
//
// With a log attached the order is rotate -> snapshot -> meta -> delete the
// retired segment, and every step of it is load-bearing:
//
//   - Rotating first means writes that land during the snapshot go to the new
//     segment, so deleting the old one cannot lose them.
//   - The meta file is written before the old segment is deleted, never after.
//     Recovery accepts a log when some retained segment carries the meta's run
//     id; if the delete came first, a crash in between would leave a meta naming
//     a run id no remaining segment has, and the engine would refuse to start on
//     a set of files that is perfectly consistent.
//
// A crash anywhere in this sequence leaves the log covering more history than
// the snapshot needs, never less. Replaying records the snapshot already
// contains is harmless: insert and delete are blind writes with no
// read-modify-write, so re-applying a range that overlaps the snapshot converges
// on the same state.
func (e *Engine) Save(path string) error {
	if e.wal == nil {
		return e.saveSnapshot(path)
	}

	run, err := newRunID()
	if err != nil {
		return err
	}

	// Under writeMu so the rotation lands at a definite point in the write
	// order rather than in the middle of a batch.
	e.writeMu.Lock()
	retired, err := e.wal.rotate(run)
	e.writeMu.Unlock()
	if err != nil {
		return err
	}

	if err := e.saveSnapshot(path); err != nil {
		return err
	}
	if err := writeMeta(metaPath(path), run); err != nil {
		return err
	}
	if err := os.Remove(retired); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("mindb: remove retired write-ahead log %s: %w", retired, err)
	}
	return nil
}

// saveSnapshot writes the engine to path atomically.
//
// The sequence is tmp -> fsync -> rename -> fsync(parent dir). That last step is
// the one almost everyone omits, and without it the rename itself can be lost:
// you fsync the data, rename over the old file, lose power, and come back to a
// directory entry that was never durably updated.
func (e *Engine) saveSnapshot(path string) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp*")
	if err != nil {
		return fmt.Errorf("mindb: create temp snapshot: %w", err)
	}
	tmpName := tmp.Name()

	// Any failure past this point must not leave the temp file behind.
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmpName)
		}
	}()

	if err = e.writeTo(tmp); err != nil {
		return err
	}
	if err = tmp.Sync(); err != nil {
		return fmt.Errorf("mindb: fsync snapshot: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("mindb: close snapshot: %w", err)
	}
	if err = os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("mindb: rename snapshot into place: %w", err)
	}
	return syncDir(dir)
}

// writeTo streams the engine under a read lock.
func (e *Engine) writeTo(f io.Writer) error {
	e.mu.RLock()
	defer e.mu.RUnlock()

	crc := crc32.NewIEEE()
	buf := bufio.NewWriterSize(f, 1<<20)
	w := io.MultiWriter(buf, crc)

	header := make([]byte, headerSize)
	copy(header, snapshotMagic[:])
	binary.LittleEndian.PutUint32(header[8:], snapshotVersion)
	binary.LittleEndian.PutUint32(header[12:], uint32(e.dims))
	binary.LittleEndian.PutUint32(header[16:], uint32(e.count))
	if _, err := w.Write(header); err != nil {
		return fmt.Errorf("mindb: write snapshot header: %w", err)
	}

	var scratch [4]byte
	vecBuf := make([]byte, e.dims*4)

	for slot := 0; slot < int(e.highWater); slot++ {
		if !e.live[slot] {
			continue
		}
		id := e.externalID[slot]
		binary.LittleEndian.PutUint32(scratch[:], uint32(len(id)))
		if _, err := w.Write(scratch[:]); err != nil {
			return err
		}
		if _, err := io.WriteString(w, id); err != nil {
			return err
		}

		pay := e.payloads[slot]
		binary.LittleEndian.PutUint32(scratch[:], uint32(len(pay)))
		if _, err := w.Write(scratch[:]); err != nil {
			return err
		}
		if len(pay) > 0 {
			if _, err := w.Write(pay); err != nil {
				return err
			}
		}

		base := slot * e.dims
		for i, v := range e.vectors[base : base+e.dims] {
			binary.LittleEndian.PutUint32(vecBuf[i*4:], stdmath.Float32bits(v))
		}
		if _, err := w.Write(vecBuf); err != nil {
			return err
		}
	}

	binary.LittleEndian.PutUint32(scratch[:], crc.Sum32())
	if _, err := buf.Write(scratch[:]); err != nil {
		return err
	}
	return buf.Flush()
}

// Load reads a snapshot into a new engine with the given capacity.
//
// Vector dimension comes from the file. Slots are reassigned on load, so a
// vector's internal index is not stable across a reload; external IDs are.
func Load(path string, capacity int) (*Engine, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	crc := crc32.NewIEEE()
	r := bufio.NewReaderSize(f, 1<<20)

	header := make([]byte, headerSize)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, fmt.Errorf("mindb: read snapshot header: %w", err)
	}
	crc.Write(header)

	if string(header[:8]) != string(snapshotMagic[:]) {
		return nil, ErrBadMagic
	}
	if v := binary.LittleEndian.Uint32(header[8:]); v != snapshotVersion {
		return nil, fmt.Errorf("%w: file is v%d, this build reads v%d", ErrBadVersion, v, snapshotVersion)
	}
	dims := int(binary.LittleEndian.Uint32(header[12:]))
	count := int(binary.LittleEndian.Uint32(header[16:]))

	if dims <= 0 {
		return nil, fmt.Errorf("mindb: snapshot declares %d dimensions", dims)
	}
	if count > capacity {
		return nil, fmt.Errorf("%w: %d vectors into capacity %d", ErrSnapshotSize, count, capacity)
	}

	e, err := New(dims, capacity)
	if err != nil {
		return nil, err
	}

	vecBuf := make([]byte, dims*4)
	for i := 0; i < count; i++ {
		id, err := readBlob(r, crc)
		if err != nil {
			return nil, fmt.Errorf("mindb: snapshot record %d: %w", i, err)
		}
		pay, err := readBlob(r, crc)
		if err != nil {
			return nil, fmt.Errorf("mindb: snapshot record %d: %w", i, err)
		}
		if _, err := io.ReadFull(r, vecBuf); err != nil {
			return nil, fmt.Errorf("mindb: snapshot record %d vector: %w", i, err)
		}
		crc.Write(vecBuf)

		// store, not Insert: the file already holds normalized vectors, and
		// re-normalizing would perturb the low bits and change search scores
		// across a reload. A fresh slice per record because store adopts it.
		vec := make([]float32, dims)
		for j := range vec {
			vec[j] = stdmath.Float32frombits(binary.LittleEndian.Uint32(vecBuf[j*4:]))
		}
		if err := e.store(string(id), vec, pay); err != nil {
			return nil, fmt.Errorf("mindb: snapshot record %d: %w", i, err)
		}
	}

	var want [4]byte
	if _, err := io.ReadFull(r, want[:]); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadChecksum, err)
	}
	if binary.LittleEndian.Uint32(want[:]) != crc.Sum32() {
		return nil, ErrBadChecksum
	}
	return e, nil
}

// readBlob reads a uint32-length-prefixed byte string, feeding the checksum.
func readBlob(r io.Reader, crc hash.Hash32) ([]byte, error) {
	var n [4]byte
	if _, err := io.ReadFull(r, n[:]); err != nil {
		return nil, err
	}
	crc.Write(n[:])

	length := binary.LittleEndian.Uint32(n[:])
	if length == 0 {
		return nil, nil
	}
	// A corrupt length field would otherwise ask for an arbitrary allocation
	// before the checksum ever gets a chance to reject the file.
	if length > 1<<28 {
		return nil, fmt.Errorf("implausible length %d", length)
	}
	b := make([]byte, length)
	if _, err := io.ReadFull(r, b); err != nil {
		return nil, err
	}
	crc.Write(b)
	return b, nil
}

// syncDir fsyncs a directory so a rename into it is durable.
//
// Windows has no directory fsync and rejects the open outright, so there the
// rename's durability is whatever the filesystem gives us. Worth knowing when
// development happens on Windows and deployment does not.
func syncDir(dir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("mindb: open snapshot dir for fsync: %w", err)
	}
	defer d.Close()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("mindb: fsync snapshot dir: %w", err)
	}
	return nil
}
