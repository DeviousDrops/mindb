package core

import (
	"runtime"

	"github.com/DeviousDrops/mindb/pkg/math"
)

// Record is a stored vector, as returned by Get.
//
// Vector and Payload are copies the caller owns outright; see Get for why that
// is not merely defensive.
type Record struct {
	ID      string
	Vector  []float32
	Payload []byte
}

// Get returns the stored record for each id, in request order.
//
// Ids that are absent — never inserted, or deleted — are omitted from the
// result rather than reported as an error, so one unknown id in a batch of a
// hundred does not fail the other ninety-nine. A caller that needs to know which
// ids missed compares the returned ids against the ones it asked for. The
// consequence worth stating: len(result) < len(ids) is normal, and the result is
// not positionally aligned with the request.
//
// # What you get back is not what you inserted
//
// Insert normalizes to unit length and discards the original norm, so Get
// returns v/|v|, not v. For cosine similarity — which is all MinDB scores — the
// two are interchangeable, but the original magnitude is not recoverable.
// Storing it would cost 4 bytes per vector and a snapshot format version, and
// nothing has needed it yet.
//
// The vector is exact, though: it is read from the float32 slab, not
// reconstructed from the int8 codes. The codes exist only to prune candidates
// during search and are never a source of truth.
//
// # Why this copies
//
// The returned slices are copies because the slab underneath is mutable and its
// slots are recycled. Handing out a sub-slice of e.vectors would let a later
// Insert — possibly to a different id that inherited the slot from the free
// list — rewrite a response the caller is still holding, with no lock left to
// protect it. One allocation per hit is the price of a result that stays valid.
func (e *Engine) Get(ids []string) []Record {
	if len(ids) == 0 {
		return nil
	}

	out := make([]Record, 0, len(ids))

	e.mu.RLock()
	defer e.mu.RUnlock()

	// One RLock for the whole batch. The alternative — locking per id — would
	// also let the engine change between lookups, so a batch could observe a
	// state no single instant ever had.
	for _, id := range ids {
		slot, ok := e.idMap[id]
		if !ok {
			continue
		}
		base := int(slot) * e.dims

		rec := Record{
			ID:     id,
			Vector: append([]float32(nil), e.vectors[base:base+e.dims]...),
		}
		if pay := e.payloads[slot]; len(pay) > 0 {
			rec.Payload = append([]byte(nil), pay...)
		}
		out = append(out, rec)
	}
	return out
}

// Stats is a point-in-time report on the engine and the kernel underneath it.
type Stats struct {
	Count    int // live vectors
	Capacity int // slots allocated at boot
	Dims     int

	// MemoryBytes is reserved, not used: allocation is eager, so this number is
	// the same the moment after New as it is at capacity. PayloadBytes is the
	// live part, since payloads are the one thing allocated on demand.
	MemoryBytes  int64
	PayloadBytes int64

	// KernelName, FastInt8 and GoArch answer "which search path is this
	// deployment actually running", which is otherwise only visible in the
	// startup log of a process nobody has the terminal for.
	KernelName string
	FastInt8   bool
	GoArch     string

	// WALHealthy is false once a log write or fsync has failed, and never goes
	// back on. It is what the readiness probe reads: the process keeps serving,
	// because in-memory state is still correct and still useful, but it stops
	// advertising itself as a place to send writes it cannot persist.
	WALEnabled bool
	WALHealthy bool
}

// Stats returns a snapshot of engine state.
func (e *Engine) Stats() Stats {
	// Read before taking the lock: the log has its own, and a faulting writer
	// may be holding this one while it waits on disk.
	walEnabled, walHealthy := e.WALEnabled(), e.WALHealthy()

	e.mu.RLock()
	defer e.mu.RUnlock()

	return Stats{
		Count:        e.count,
		Capacity:     e.capacity,
		Dims:         e.dims,
		MemoryBytes:  e.memoryBytes(),
		PayloadBytes: e.payloadBytes,
		KernelName:   math.KernelName(),
		FastInt8:     math.HasFastInt8(),
		GoArch:       runtime.GOARCH,
		WALEnabled:   walEnabled,
		WALHealthy:   walHealthy,
	}
}

// memoryBytes is the eagerly reserved footprint of the slab arrays.
//
// Per vector: dims*4 for the float32 copy, dims*1 for the int8 code, and 4 each
// for the scale and the residual norm. Payloads, ids and the id map are on top
// of this and vary; PayloadBytes covers the largest of them.
//
// Caller must hold at least the read lock.
func (e *Engine) memoryBytes() int64 {
	return int64(e.capacity) * (int64(e.dims)*5 + 8)
}
