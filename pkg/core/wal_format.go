package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	stdmath "math"
)

// Write-ahead log format. All integers little-endian.
//
//	segment header
//	  magic    8   "MINDBWAL"
//	  version  uint32
//	  dims     uint32   must match the engine; a mismatch refuses to start
//	  run id   16       ties this segment to the snapshot it follows
//
//	record
//	  len      uint32   length of body
//	  crc      uint32   crc32-IEEE over the four len bytes and the body
//	  body     len bytes
//
//	body, insert (op = 1)
//	  op       uint8
//	  idLen    uint32, id
//	  payLen   uint32, payload
//	  values   dims * float32, unit-normalized exactly as stored
//
//	body, delete (op = 2)
//	  op       uint8
//	  idLen    uint32, id
//
// The checksum covers the length field, not just the body. A corrupted length
// otherwise sends the reader off to consume an arbitrary number of bytes, and
// the record it then checksums is not the one that was damaged — the failure
// gets reported at the wrong offset, after an implausible allocation on the way.
//
// Vectors are logged already normalized, matching the snapshot, so replay can go
// through store rather than Insert. Re-running Normalize on a unit vector is not
// the identity in float32, so logging the caller's vector instead would make a
// recovered engine return subtly different scores from the process that crashed.
const (
	walVersion      = 1
	walHeaderSize   = 8 + 4 + 4 + runIDLen
	walRecordHeader = 4 + 4

	// walMaxRecord bounds what a length field can ask us to allocate before the
	// checksum has had any chance to reject it.
	walMaxRecord = 1 << 28
)

var walMagic = [8]byte{'M', 'I', 'N', 'D', 'B', 'W', 'A', 'L'}

var (
	ErrWALBadMagic   = errors.New("mindb: not a MinDB write-ahead log")
	ErrWALBadVersion = errors.New("mindb: unsupported write-ahead log version")
	ErrWALDimsChange = errors.New("mindb: write-ahead log dimension does not match engine")
)

// walOp is the operation a record replays.
type walOp uint8

const (
	opInsert walOp = 1
	opDelete walOp = 2
)

// walRecord is one decoded log entry. Vector and Payload alias the decode
// buffer; replay adopts the vector and copies the payload, so nothing else may
// retain either.
type walRecord struct {
	op      walOp
	id      string
	vector  []float32
	payload []byte
}

// appendSegmentHeader writes the fixed header that opens every segment.
func appendSegmentHeader(dst []byte, dims int, run runID) []byte {
	dst = append(dst, walMagic[:]...)
	dst = binary.LittleEndian.AppendUint32(dst, walVersion)
	dst = binary.LittleEndian.AppendUint32(dst, uint32(dims))
	return append(dst, run[:]...)
}

// parseSegmentHeader validates a segment header against the engine it is about
// to be replayed into.
func parseSegmentHeader(b []byte, dims int) (runID, error) {
	var run runID
	if len(b) < walHeaderSize {
		return run, fmt.Errorf("mindb: write-ahead log header is %d bytes, want %d", len(b), walHeaderSize)
	}
	if string(b[:8]) != string(walMagic[:]) {
		return run, ErrWALBadMagic
	}
	if v := binary.LittleEndian.Uint32(b[8:]); v != walVersion {
		return run, fmt.Errorf("%w: file is v%d, this build reads v%d", ErrWALBadVersion, v, walVersion)
	}
	if d := int(binary.LittleEndian.Uint32(b[12:])); d != dims {
		return run, fmt.Errorf("%w: log is %d, engine is %d", ErrWALDimsChange, d, dims)
	}
	copy(run[:], b[16:])
	return run, nil
}

// appendInsert frames an insert record onto dst.
//
// vec must already be normalized and payload may be empty.
func appendInsert(dst []byte, id string, vec []float32, payload []byte) []byte {
	body := make([]byte, 0, 1+4+len(id)+4+len(payload)+len(vec)*4)
	body = append(body, byte(opInsert))
	body = binary.LittleEndian.AppendUint32(body, uint32(len(id)))
	body = append(body, id...)
	body = binary.LittleEndian.AppendUint32(body, uint32(len(payload)))
	body = append(body, payload...)
	for _, v := range vec {
		body = binary.LittleEndian.AppendUint32(body, stdmath.Float32bits(v))
	}
	return frame(dst, body)
}

// appendDelete frames a delete record onto dst.
func appendDelete(dst []byte, id string) []byte {
	body := make([]byte, 0, 1+4+len(id))
	body = append(body, byte(opDelete))
	body = binary.LittleEndian.AppendUint32(body, uint32(len(id)))
	body = append(body, id...)
	return frame(dst, body)
}

// frame prefixes body with its length and a checksum over both.
func frame(dst, body []byte) []byte {
	var hdr [walRecordHeader]byte
	binary.LittleEndian.PutUint32(hdr[:4], uint32(len(body)))

	crc := crc32.NewIEEE()
	crc.Write(hdr[:4])
	crc.Write(body)
	binary.LittleEndian.PutUint32(hdr[4:], crc.Sum32())

	dst = append(dst, hdr[:]...)
	return append(dst, body...)
}

// errBadRecord marks a record that failed its checksum or ran off the end of the
// file. Replay stops there rather than propagating it: a torn tail is the
// expected result of killing a process mid-append, not corruption.
var errBadRecord = errors.New("mindb: damaged write-ahead log record")

// decodeBody turns a verified record body into a replayable operation.
//
// The returned vector aliases body; the caller owns body and must not reuse it
// while the record is live.
func decodeBody(body []byte, dims int) (walRecord, error) {
	var rec walRecord
	if len(body) < 1 {
		return rec, errBadRecord
	}
	rec.op = walOp(body[0])
	rest := body[1:]

	id, rest, ok := takeBlob(rest)
	if !ok {
		return rec, errBadRecord
	}
	rec.id = string(id)

	switch rec.op {
	case opDelete:
		// A trailing byte means the record is not what it claims to be, even
		// though the checksum passed — which happens when a stale record of a
		// different shape lands at this offset.
		if len(rest) != 0 {
			return rec, errBadRecord
		}
		if rec.id == "" {
			return rec, errBadRecord
		}
		return rec, nil

	case opInsert:
		pay, rest, ok := takeBlob(rest)
		if !ok {
			return rec, errBadRecord
		}
		if len(rest) != dims*4 {
			return rec, errBadRecord
		}
		if rec.id == "" {
			return rec, errBadRecord
		}
		rec.payload = pay
		rec.vector = make([]float32, dims)
		for i := range rec.vector {
			rec.vector[i] = stdmath.Float32frombits(binary.LittleEndian.Uint32(rest[i*4:]))
		}
		return rec, nil

	default:
		return rec, errBadRecord
	}
}

// takeBlob splits a uint32-length-prefixed byte string off the front of b.
func takeBlob(b []byte) (blob, rest []byte, ok bool) {
	if len(b) < 4 {
		return nil, nil, false
	}
	n := int(binary.LittleEndian.Uint32(b[:4]))
	if n < 0 || n > len(b)-4 {
		return nil, nil, false
	}
	return b[4 : 4+n], b[4+n:], true
}
