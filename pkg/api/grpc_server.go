// Package api implements the FlatBuffers gRPC surface over the core engine.
package api

import (
	"context"
	"encoding/binary"
	"errors"
	"unsafe"

	flatbuffers "github.com/google/flatbuffers/go"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/typicallhavok/mindb/pkg/core"
	"github.com/typicallhavok/mindb/pkg/mindb"
)

// Server implements mindb.VectorServiceServer.
type Server struct {
	// Embedded so that adding an RPC to the schema does not break this type at
	// compile time. The trade-off is real: a handler that is never written now
	// returns Unimplemented at runtime instead of failing the build.
	mindb.UnimplementedVectorServiceServer

	engine       *core.Engine
	snapshotPath string
}

// New returns a server over engine. snapshotPath may be empty, in which case
// the Snapshot RPC reports that persistence is disabled rather than failing.
func New(engine *core.Engine, snapshotPath string) *Server {
	return &Server{engine: engine, snapshotPath: snapshotPath}
}

// Insert stores every vector in the request.
//
// Not atomic: on the first rejected vector the RPC fails, and vectors earlier in
// the batch are already stored. The engine has no transaction boundary to roll
// back to, and pretending otherwise would be worse than documenting it.
func (s *Server) Insert(_ context.Context, req *mindb.InsertRequest) (*flatbuffers.Builder, error) {
	n := req.VectorsLength()
	var vec mindb.Vector
	var inserted int32

	for i := 0; i < n; i++ {
		if !req.Vectors(&vec, i) {
			return nil, status.Errorf(codes.InvalidArgument, "vector %d is malformed", i)
		}
		values := float32Vector(vec.Table(), vectorValuesSlot)
		if len(values) == 0 {
			return nil, status.Errorf(codes.InvalidArgument, "vector %d has no values", i)
		}
		if err := s.engine.Insert(string(vec.Id()), values, vec.PayloadBytes()); err != nil {
			return nil, insertError(i, inserted, err)
		}
		inserted++
	}

	b := flatbuffers.NewBuilder(32)
	mindb.InsertResponseStart(b)
	mindb.InsertResponseAddInsertedCount(b, inserted)
	b.Finish(mindb.InsertResponseEnd(b))
	return b, nil
}

func insertError(i int, inserted int32, err error) error {
	code := codes.Internal
	switch {
	case errors.Is(err, core.ErrDimensionMismatch),
		errors.Is(err, core.ErrZeroVector),
		errors.Is(err, core.ErrEmptyID):
		code = codes.InvalidArgument
	case errors.Is(err, core.ErrCapacityExceeded):
		code = codes.ResourceExhausted
	}
	return status.Errorf(code, "vector %d: %v (%d earlier vectors in this batch were stored)", i, err, inserted)
}

// Search returns the top_k nearest vectors to query_vector.
func (s *Server) Search(_ context.Context, req *mindb.SearchRequest) (*flatbuffers.Builder, error) {
	query := float32Vector(req.Table(), searchQueryVectorSlot)
	if len(query) == 0 {
		return nil, status.Error(codes.InvalidArgument, "query_vector is empty")
	}

	results, err := s.engine.Search(query, int(req.TopK()))
	if err != nil {
		if errors.Is(err, core.ErrDimensionMismatch) || errors.Is(err, core.ErrZeroVector) {
			return nil, status.Error(codes.InvalidArgument, err.Error())
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	// Builders are deliberately not pooled. gRPC marshals the returned builder
	// after this function returns, so there is no point at which we could
	// safely recycle it. This is the half of "zero-copy FlatBuffers" that is
	// not real: reading the request avoids a copy, building the response does
	// not.
	b := flatbuffers.NewBuilder(256 + len(results)*96)

	offsets := make([]flatbuffers.UOffsetT, len(results))
	for i, r := range results {
		id := b.CreateString(r.ID)
		var payload flatbuffers.UOffsetT
		if len(r.Payload) > 0 {
			payload = b.CreateByteVector(r.Payload)
		}
		mindb.SearchResultStart(b)
		mindb.SearchResultAddId(b, id)
		mindb.SearchResultAddScore(b, r.Score)
		if payload != 0 {
			mindb.SearchResultAddPayload(b, payload)
		}
		offsets[i] = mindb.SearchResultEnd(b)
	}

	// Vectors build back to front, so results must be prepended in reverse to
	// come out in rank order on the wire.
	mindb.SearchResponseStartResultsVector(b, len(offsets))
	for i := len(offsets) - 1; i >= 0; i-- {
		b.PrependUOffsetT(offsets[i])
	}
	vec := b.EndVector(len(offsets))

	mindb.SearchResponseStart(b)
	mindb.SearchResponseAddResults(b, vec)
	b.Finish(mindb.SearchResponseEnd(b))
	return b, nil
}

// Delete removes every id in the request, reporting how many existed.
//
// Like Insert, this is not atomic: a durability failure part way through leaves
// the ids before it deleted. The error says how many that was, because a caller
// that retries the whole batch needs to know the first attempt was not a no-op.
func (s *Server) Delete(_ context.Context, req *mindb.DeleteRequest) (*flatbuffers.Builder, error) {
	var deleted int32
	for i := 0; i < req.IdsLength(); i++ {
		existed, err := s.engine.Delete(string(req.Ids(i)))
		if err != nil {
			return nil, status.Errorf(codes.Internal,
				"delete %q: %v (%d of %d ids already deleted)", req.Ids(i), err, deleted, req.IdsLength())
		}
		if existed {
			deleted++
		}
	}

	b := flatbuffers.NewBuilder(32)
	mindb.DeleteResponseStart(b)
	mindb.DeleteResponseAddDeletedCount(b, deleted)
	b.Finish(mindb.DeleteResponseEnd(b))
	return b, nil
}

// Snapshot writes the engine to disk atomically.
func (s *Server) Snapshot(_ context.Context, _ *mindb.SnapshotRequest) (*flatbuffers.Builder, error) {
	ok, msg := true, "snapshot written"
	if s.snapshotPath == "" {
		ok, msg = false, "snapshots are disabled: server started without -snapshot"
	} else if err := s.engine.Save(s.snapshotPath); err != nil {
		ok, msg = false, err.Error()
	}

	b := flatbuffers.NewBuilder(128)
	m := b.CreateString(msg)
	mindb.SnapshotResponseStart(b)
	mindb.SnapshotResponseAddSuccess(b, ok)
	mindb.SnapshotResponseAddMessage(b, m)
	b.Finish(mindb.SnapshotResponseEnd(b))
	return b, nil
}

// Get returns the stored record for each requested id, in request order.
//
// Ids that are absent -- never inserted, or deleted -- are omitted from the
// response rather than reported as an error, so one unknown id cannot fail a
// batch. The response is therefore not positionally aligned with the request;
// callers match on id.
//
// The vectors returned are unit-normalized, because that is what Insert stored:
// the original magnitude is discarded at insert and is not recoverable here.
func (s *Server) Get(_ context.Context, req *mindb.GetRequest) (*flatbuffers.Builder, error) {
	n := req.IdsLength()
	ids := make([]string, n)
	for i := 0; i < n; i++ {
		ids[i] = string(req.Ids(i))
	}

	recs := s.engine.Get(ids)

	size := 256
	if len(recs) > 0 {
		size += len(recs) * (96 + len(recs[0].Vector)*4)
	}
	b := flatbuffers.NewBuilder(size)

	offsets := make([]flatbuffers.UOffsetT, len(recs))
	for i, rec := range recs {
		// Strings and vectors have to be built before the table that refers to
		// them; FlatBuffers forbids creating either inside an open table.
		id := b.CreateString(rec.ID)
		var payload flatbuffers.UOffsetT
		if len(rec.Payload) > 0 {
			payload = b.CreateByteVector(rec.Payload)
		}

		mindb.VectorStartValuesVector(b, len(rec.Vector))
		for j := len(rec.Vector) - 1; j >= 0; j-- {
			b.PrependFloat32(rec.Vector[j])
		}
		values := b.EndVector(len(rec.Vector))

		mindb.VectorStart(b)
		mindb.VectorAddId(b, id)
		mindb.VectorAddValues(b, values)
		if payload != 0 {
			mindb.VectorAddPayload(b, payload)
		}
		offsets[i] = mindb.VectorEnd(b)
	}

	// Vectors build back to front, so records are prepended in reverse to come
	// out in request order on the wire.
	mindb.GetResponseStartVectorsVector(b, len(offsets))
	for i := len(offsets) - 1; i >= 0; i-- {
		b.PrependUOffsetT(offsets[i])
	}
	vec := b.EndVector(len(offsets))

	mindb.GetResponseStart(b)
	mindb.GetResponseAddVectors(b, vec)
	b.Finish(mindb.GetResponseEnd(b))
	return b, nil
}

// Stats reports engine counters and which search kernel is live.
func (s *Server) Stats(_ context.Context, _ *mindb.StatsRequest) (*flatbuffers.Builder, error) {
	st := s.engine.Stats()

	b := flatbuffers.NewBuilder(256)
	kernel := b.CreateString(st.KernelName)
	goarch := b.CreateString(st.GoArch)

	mindb.StatsResponseStart(b)
	mindb.StatsResponseAddVectorCount(b, uint32(st.Count))
	mindb.StatsResponseAddCapacity(b, uint32(st.Capacity))
	mindb.StatsResponseAddDims(b, uint32(st.Dims))
	mindb.StatsResponseAddMemoryBytes(b, uint64(st.MemoryBytes))
	mindb.StatsResponseAddPayloadBytes(b, uint64(st.PayloadBytes))
	mindb.StatsResponseAddKernelName(b, kernel)
	mindb.StatsResponseAddFastInt8(b, st.FastInt8)
	mindb.StatsResponseAddGoarch(b, goarch)
	b.Finish(mindb.StatsResponseEnd(b))
	return b, nil
}

// FlatBuffers vtable slots, matching field order in fbs/mindb.fbs. The generated
// accessors hardcode these same numbers; they are repeated here because the
// zero-copy read below bypasses those accessors.
const (
	vectorValuesSlot      flatbuffers.VOffsetT = 6 // Vector.values, second field
	searchQueryVectorSlot flatbuffers.VOffsetT = 4 // SearchRequest.query_vector, first field
)

// nativeLittleEndian reports whether this platform's float32 layout matches the
// FlatBuffers wire format, which is little-endian. Every platform Go targets in
// practice is, but the unsafe reinterpret below is silently wrong if not, so it
// is checked rather than assumed.
var nativeLittleEndian = func() bool {
	var probe uint32 = 1
	return (*[4]byte)(unsafe.Pointer(&probe))[0] == 1
}()

// float32Vector returns the float32 vector at the given vtable slot without
// copying it.
//
// This is the real zero-copy win the design is after. The generated accessor
// reads one element per call through bounds-checked offset arithmetic; the
// buffer already holds little-endian 4-byte-aligned float32, so it can simply be
// reinterpreted in place.
//
// The returned slice aliases the request buffer, which gRPC recycles once the
// handler returns. Callers must not retain it — core.Engine copies on Insert
// and on Search for exactly this reason.
func float32Vector(tab flatbuffers.Table, slot flatbuffers.VOffsetT) []float32 {
	o := flatbuffers.UOffsetT(tab.Offset(slot))
	if o == 0 {
		return nil
	}
	n := tab.VectorLen(o)
	if n == 0 {
		return nil
	}
	start := tab.Vector(o)

	if !nativeLittleEndian {
		out := make([]float32, n)
		for i := range out {
			bits := binary.LittleEndian.Uint32(tab.Bytes[start+flatbuffers.UOffsetT(i*4):])
			out[i] = *(*float32)(unsafe.Pointer(&bits))
		}
		return out
	}

	// Bounds-check the whole region once, so the reinterpret cannot run off the
	// end of a truncated or hostile buffer.
	end := int(start) + n*4
	if end > len(tab.Bytes) {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&tab.Bytes[start])), n)
}
