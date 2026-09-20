package api

import (
	"context"
	"fmt"
	stdmath "math"
	"math/rand"
	"net"
	"path/filepath"
	"testing"
	"time"

	flatbuffers "github.com/google/flatbuffers/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"github.com/DeviousDrops/mindb/pkg/core"
	"github.com/DeviousDrops/mindb/pkg/mindb"
)

const testDims = 16

// buildInsert encodes vectors into an InsertRequest buffer.
func buildInsert(vectors map[string][]float32, payloads map[string][]byte) *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(0)

	ids := make([]string, 0, len(vectors))
	for id := range vectors {
		ids = append(ids, id)
	}

	offsets := make([]flatbuffers.UOffsetT, 0, len(ids))
	for _, id := range ids {
		vals := vectors[id]
		idOff := b.CreateString(id)

		mindb.VectorStartValuesVector(b, len(vals))
		for i := len(vals) - 1; i >= 0; i-- {
			b.PrependFloat32(vals[i])
		}
		valsOff := b.EndVector(len(vals))

		var payOff flatbuffers.UOffsetT
		if p := payloads[id]; len(p) > 0 {
			payOff = b.CreateByteVector(p)
		}

		mindb.VectorStart(b)
		mindb.VectorAddId(b, idOff)
		mindb.VectorAddValues(b, valsOff)
		if payOff != 0 {
			mindb.VectorAddPayload(b, payOff)
		}
		offsets = append(offsets, mindb.VectorEnd(b))
	}

	mindb.InsertRequestStartVectorsVector(b, len(offsets))
	for i := len(offsets) - 1; i >= 0; i-- {
		b.PrependUOffsetT(offsets[i])
	}
	vecs := b.EndVector(len(offsets))

	mindb.InsertRequestStart(b)
	mindb.InsertRequestAddVectors(b, vecs)
	b.Finish(mindb.InsertRequestEnd(b))
	return b
}

func buildSearch(query []float32, topK int32) *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(0)
	mindb.SearchRequestStartQueryVectorVector(b, len(query))
	for i := len(query) - 1; i >= 0; i-- {
		b.PrependFloat32(query[i])
	}
	qv := b.EndVector(len(query))

	mindb.SearchRequestStart(b)
	mindb.SearchRequestAddQueryVector(b, qv)
	mindb.SearchRequestAddTopK(b, topK)
	b.Finish(mindb.SearchRequestEnd(b))
	return b
}

func buildDelete(ids []string) *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(0)
	offsets := make([]flatbuffers.UOffsetT, len(ids))
	for i, id := range ids {
		offsets[i] = b.CreateString(id)
	}
	mindb.DeleteRequestStartIdsVector(b, len(offsets))
	for i := len(offsets) - 1; i >= 0; i-- {
		b.PrependUOffsetT(offsets[i])
	}
	vec := b.EndVector(len(offsets))

	mindb.DeleteRequestStart(b)
	mindb.DeleteRequestAddIds(b, vec)
	b.Finish(mindb.DeleteRequestEnd(b))
	return b
}

func buildSnapshot() *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(0)
	mindb.SnapshotRequestStart(b)
	b.Finish(mindb.SnapshotRequestEnd(b))
	return b
}

// startServer boots a real gRPC server on a loopback port and returns a client.
// Nothing is mocked: the FlatBuffers codec is exactly the thing most likely to
// be misconfigured, and it only fails at request time.
func startServer(t *testing.T, engine *core.Engine, snapshotPath string) mindb.VectorServiceClient {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	srv := grpc.NewServer(grpc.ForceServerCodec(flatbuffers.FlatbuffersCodec{}))
	mindb.RegisterVectorServiceServer(srv, New(engine, snapshotPath))
	go srv.Serve(lis)
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(flatbuffers.FlatbuffersCodec{})),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })

	return mindb.NewVectorServiceClient(conn)
}

func ctxWithTimeout(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestEndToEnd(t *testing.T) {
	engine, err := core.New(testDims, 100)
	if err != nil {
		t.Fatal(err)
	}
	snapPath := filepath.Join(t.TempDir(), "snap.mdb")
	client := startServer(t, engine, snapPath)
	ctx := ctxWithTimeout(t)

	// Two vectors pointing along different axes, so the expected ranking is
	// obvious rather than something the test has to recompute.
	target := make([]float32, testDims)
	target[0] = 1
	other := make([]float32, testDims)
	other[1] = 1

	vectors := map[string][]float32{"target": target, "other": other}
	payloads := map[string][]byte{"target": []byte(`{"kind":"target"}`)}

	insertResp, err := client.Insert(ctx, buildInsert(vectors, payloads))
	if err != nil {
		t.Fatalf("Insert failed (a missing codec shows up exactly here): %v", err)
	}
	if insertResp.InsertedCount() != 2 {
		t.Fatalf("inserted %d, want 2", insertResp.InsertedCount())
	}

	searchResp, err := client.Search(ctx, buildSearch(target, 2))
	if err != nil {
		t.Fatal(err)
	}
	if got := searchResp.ResultsLength(); got != 2 {
		t.Fatalf("got %d results, want 2", got)
	}

	var hit mindb.SearchResult
	if !searchResp.Results(&hit, 0) {
		t.Fatal("could not read result 0")
	}
	if string(hit.Id()) != "target" {
		t.Errorf("rank 0 is %q, want \"target\"", hit.Id())
	}
	if stdmath.Abs(float64(hit.Score()-1)) > 1e-6 {
		t.Errorf("rank 0 score %v, want 1", hit.Score())
	}
	if string(hit.PayloadBytes()) != `{"kind":"target"}` {
		t.Errorf("payload round-trip failed: %q", hit.PayloadBytes())
	}

	if !searchResp.Results(&hit, 1) {
		t.Fatal("could not read result 1")
	}
	if string(hit.Id()) != "other" {
		t.Errorf("rank 1 is %q, want \"other\"", hit.Id())
	}

	snapResp, err := client.Snapshot(ctx, buildSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	if !snapResp.Success() {
		t.Fatalf("snapshot failed: %s", snapResp.Message())
	}
	if reloaded, err := core.Load(snapPath, 100); err != nil {
		t.Fatalf("snapshot written over gRPC does not load: %v", err)
	} else if reloaded.Len() != 2 {
		t.Errorf("reloaded %d vectors, want 2", reloaded.Len())
	}

	delResp, err := client.Delete(ctx, buildDelete([]string{"target", "missing"}))
	if err != nil {
		t.Fatal(err)
	}
	if delResp.DeletedCount() != 1 {
		t.Errorf("deleted %d, want 1 (absent ids must not count)", delResp.DeletedCount())
	}

	searchResp, err = client.Search(ctx, buildSearch(target, 5))
	if err != nil {
		t.Fatal(err)
	}
	if got := searchResp.ResultsLength(); got != 1 {
		t.Fatalf("after delete got %d results, want 1", got)
	}
}

func TestSnapshotDisabledReportsFailure(t *testing.T) {
	engine, _ := core.New(testDims, 10)
	client := startServer(t, engine, "")

	resp, err := client.Snapshot(ctxWithTimeout(t), buildSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Success() {
		t.Error("snapshot reported success with persistence disabled")
	}
	if len(resp.Message()) == 0 {
		t.Error("no message explaining why the snapshot failed")
	}
}

func TestInsertErrorsMapToStatusCodes(t *testing.T) {
	engine, _ := core.New(testDims, 1)
	client := startServer(t, engine, "")
	ctx := ctxWithTimeout(t)

	t.Run("dimension mismatch", func(t *testing.T) {
		bad := map[string][]float32{"a": {1, 2, 3}}
		if _, err := client.Insert(ctx, buildInsert(bad, nil)); status.Code(err) != codes.InvalidArgument {
			t.Errorf("got %v, want InvalidArgument", status.Code(err))
		}
	})

	t.Run("zero vector", func(t *testing.T) {
		zero := map[string][]float32{"z": make([]float32, testDims)}
		if _, err := client.Insert(ctx, buildInsert(zero, nil)); status.Code(err) != codes.InvalidArgument {
			t.Errorf("got %v, want InvalidArgument", status.Code(err))
		}
	})

	t.Run("capacity exceeded", func(t *testing.T) {
		first := make([]float32, testDims)
		first[0] = 1
		if _, err := client.Insert(ctx, buildInsert(map[string][]float32{"one": first}, nil)); err != nil {
			t.Fatal(err)
		}
		second := make([]float32, testDims)
		second[1] = 1
		_, err := client.Insert(ctx, buildInsert(map[string][]float32{"two": second}, nil))
		if status.Code(err) != codes.ResourceExhausted {
			t.Errorf("got %v, want ResourceExhausted", status.Code(err))
		}
	})
}

func TestSearchRejectsEmptyQuery(t *testing.T) {
	engine, _ := core.New(testDims, 10)
	client := startServer(t, engine, "")

	b := flatbuffers.NewBuilder(0)
	mindb.SearchRequestStart(b)
	mindb.SearchRequestAddTopK(b, 5)
	b.Finish(mindb.SearchRequestEnd(b))

	if _, err := client.Search(ctxWithTimeout(t), b); status.Code(err) != codes.InvalidArgument {
		t.Errorf("got %v, want InvalidArgument", status.Code(err))
	}
}

// TestZeroCopyReadMatchesGeneratedAccessor is the safety net for the unsafe
// reinterpret. If the two ever disagree, every score the server returns is
// wrong in a way no other test would notice.
func TestZeroCopyReadMatchesGeneratedAccessor(t *testing.T) {
	r := rand.New(rand.NewSource(42))

	for _, dims := range []int{1, 2, 3, 7, 8, 16, 100, 768} {
		vals := make([]float32, dims)
		for i := range vals {
			vals[i] = r.Float32()*2 - 1
		}

		t.Run(fmt.Sprintf("SearchRequest/dims=%d", dims), func(t *testing.T) {
			buf := buildSearch(vals, 5).FinishedBytes()
			req := mindb.GetRootAsSearchRequest(buf, 0)

			got := float32Vector(req.Table(), searchQueryVectorSlot)
			if len(got) != dims {
				t.Fatalf("zero-copy read gave %d elements, want %d", len(got), dims)
			}
			for i := range got {
				if want := req.QueryVector(i); got[i] != want {
					t.Fatalf("element %d: zero-copy %v, generated accessor %v", i, got[i], want)
				}
			}
		})

		t.Run(fmt.Sprintf("Vector/dims=%d", dims), func(t *testing.T) {
			buf := buildInsert(map[string][]float32{"a": vals}, nil).FinishedBytes()
			req := mindb.GetRootAsInsertRequest(buf, 0)

			var vec mindb.Vector
			if !req.Vectors(&vec, 0) {
				t.Fatal("could not read vector 0")
			}
			got := float32Vector(vec.Table(), vectorValuesSlot)
			if len(got) != dims {
				t.Fatalf("zero-copy read gave %d elements, want %d", len(got), dims)
			}
			for i := range got {
				if want := vec.Values(i); got[i] != want {
					t.Fatalf("element %d: zero-copy %v, generated accessor %v", i, got[i], want)
				}
			}
		})
	}
}

func TestFloat32VectorHandlesMissingField(t *testing.T) {
	b := flatbuffers.NewBuilder(0)
	mindb.SearchRequestStart(b)
	mindb.SearchRequestAddTopK(b, 5)
	b.Finish(mindb.SearchRequestEnd(b))

	req := mindb.GetRootAsSearchRequest(b.FinishedBytes(), 0)
	if got := float32Vector(req.Table(), searchQueryVectorSlot); got != nil {
		t.Errorf("absent vector field returned %v, want nil", got)
	}
}

func buildGet(ids []string) *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(0)
	offsets := make([]flatbuffers.UOffsetT, len(ids))
	for i, id := range ids {
		offsets[i] = b.CreateString(id)
	}
	mindb.GetRequestStartIdsVector(b, len(offsets))
	for i := len(offsets) - 1; i >= 0; i-- {
		b.PrependUOffsetT(offsets[i])
	}
	vec := b.EndVector(len(offsets))

	mindb.GetRequestStart(b)
	mindb.GetRequestAddIds(b, vec)
	b.Finish(mindb.GetRequestEnd(b))
	return b
}

func buildStats() *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(0)
	mindb.StatsRequestStart(b)
	b.Finish(mindb.StatsRequestEnd(b))
	return b
}

// getIDs reads the ids out of a GetResponse in wire order, which is the thing
// the response ordering guarantee is actually about.
func getIDs(resp *mindb.GetResponse) []string {
	out := make([]string, resp.VectorsLength())
	var v mindb.Vector
	for i := range out {
		resp.Vectors(&v, i)
		out[i] = string(v.Id())
	}
	return out
}

func TestGetRoundTripsVectorAndPayload(t *testing.T) {
	engine, err := core.New(testDims, 100)
	if err != nil {
		t.Fatal(err)
	}
	client := startServer(t, engine, "")
	ctx := ctxWithTimeout(t)

	// Already unit length, so the response should match byte for byte: this test
	// is about the wire path, not about normalization.
	vec := make([]float32, testDims)
	vec[3] = 1
	payload := []byte(`{"kind":"round-trip"}`)

	if _, err := client.Insert(ctx, buildInsert(
		map[string][]float32{"a": vec},
		map[string][]byte{"a": payload},
	)); err != nil {
		t.Fatal(err)
	}

	resp, err := client.Get(ctx, buildGet([]string{"a"}))
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.VectorsLength(); got != 1 {
		t.Fatalf("got %d vectors, want 1", got)
	}

	var v mindb.Vector
	if !resp.Vectors(&v, 0) {
		t.Fatal("could not read vector 0")
	}
	if string(v.Id()) != "a" {
		t.Errorf("id %q, want \"a\"", v.Id())
	}
	if got := v.ValuesLength(); got != testDims {
		t.Fatalf("got %d values, want %d", got, testDims)
	}
	for i := 0; i < testDims; i++ {
		if stdmath.Abs(float64(v.Values(i)-vec[i])) > 1e-6 {
			t.Errorf("values[%d] = %v, want %v", i, v.Values(i), vec[i])
		}
	}
	if string(v.PayloadBytes()) != string(payload) {
		t.Errorf("payload %q, want %q", v.PayloadBytes(), payload)
	}
}

func TestGetOmitsMissingIDsAndKeepsOrder(t *testing.T) {
	engine, err := core.New(testDims, 100)
	if err != nil {
		t.Fatal(err)
	}
	client := startServer(t, engine, "")
	ctx := ctxWithTimeout(t)

	vectors := map[string][]float32{}
	for _, id := range []string{"a", "b", "c"} {
		v := make([]float32, testDims)
		v[0] = 1
		vectors[id] = v
	}
	if _, err := client.Insert(ctx, buildInsert(vectors, nil)); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Delete(ctx, buildDelete([]string{"b"})); err != nil {
		t.Fatal(err)
	}

	// "b" was deleted and "nope" never existed: both must vanish rather than
	// fail the batch, and the survivors must stay in request order.
	resp, err := client.Get(ctx, buildGet([]string{"c", "nope", "a", "b"}))
	if err != nil {
		t.Fatalf("a missing id must not be an error: %v", err)
	}
	got := getIDs(resp)
	want := []string{"c", "a"}
	if len(got) != len(want) {
		t.Fatalf("got ids %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got ids %v, want %v", got, want)
		}
	}
}

func TestGetEmptyRequestReturnsEmptyResponse(t *testing.T) {
	engine, err := core.New(testDims, 100)
	if err != nil {
		t.Fatal(err)
	}
	client := startServer(t, engine, "")

	// An empty vector and an absent one decode identically here, so this pins
	// only what callers can rely on: no error, nothing returned.
	resp, err := client.Get(ctxWithTimeout(t), buildGet(nil))
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.VectorsLength(); got != 0 {
		t.Errorf("got %d vectors, want 0", got)
	}
}

func TestGetOmitsPayloadWhenThereIsNone(t *testing.T) {
	engine, err := core.New(testDims, 100)
	if err != nil {
		t.Fatal(err)
	}
	client := startServer(t, engine, "")
	ctx := ctxWithTimeout(t)

	v := make([]float32, testDims)
	v[0] = 1
	if _, err := client.Insert(ctx, buildInsert(map[string][]float32{"bare": v}, nil)); err != nil {
		t.Fatal(err)
	}

	resp, err := client.Get(ctx, buildGet([]string{"bare"}))
	if err != nil {
		t.Fatal(err)
	}
	var got mindb.Vector
	if !resp.Vectors(&got, 0) {
		t.Fatal("could not read vector 0")
	}
	if n := got.PayloadLength(); n != 0 {
		t.Errorf("payload length %d, want 0", n)
	}
}

func TestStatsReportsEngineAndKernel(t *testing.T) {
	engine, err := core.New(testDims, 64)
	if err != nil {
		t.Fatal(err)
	}
	client := startServer(t, engine, "")
	ctx := ctxWithTimeout(t)

	v := make([]float32, testDims)
	v[0] = 1
	payload := []byte("0123456789")
	if _, err := client.Insert(ctx, buildInsert(
		map[string][]float32{"a": v},
		map[string][]byte{"a": payload},
	)); err != nil {
		t.Fatal(err)
	}

	resp, err := client.Stats(ctx, buildStats())
	if err != nil {
		t.Fatal(err)
	}

	want := engine.Stats()
	if resp.VectorCount() != uint32(want.Count) {
		t.Errorf("vector_count = %d, want %d", resp.VectorCount(), want.Count)
	}
	if resp.Capacity() != uint32(want.Capacity) {
		t.Errorf("capacity = %d, want %d", resp.Capacity(), want.Capacity)
	}
	if resp.Dims() != uint32(want.Dims) {
		t.Errorf("dims = %d, want %d", resp.Dims(), want.Dims)
	}
	if resp.MemoryBytes() != uint64(want.MemoryBytes) {
		t.Errorf("memory_bytes = %d, want %d", resp.MemoryBytes(), want.MemoryBytes)
	}
	if resp.PayloadBytes() != uint64(len(payload)) {
		t.Errorf("payload_bytes = %d, want %d", resp.PayloadBytes(), len(payload))
	}

	// The whole point of exposing these is that a deployment can report which
	// search path is live, so they must reach the wire non-empty.
	if string(resp.KernelName()) != want.KernelName || len(resp.KernelName()) == 0 {
		t.Errorf("kernel_name = %q, want %q", resp.KernelName(), want.KernelName)
	}
	if resp.FastInt8() != want.FastInt8 {
		t.Errorf("fast_int8 = %t, want %t", resp.FastInt8(), want.FastInt8)
	}
	if string(resp.Goarch()) != want.GoArch {
		t.Errorf("goarch = %q, want %q", resp.Goarch(), want.GoArch)
	}

	// An engine with no log reports healthy, because there is nothing that can
	// have failed. Readiness has to read wal_enabled first.
	if resp.WalEnabled() {
		t.Error("wal_enabled = true, want false for an engine with no log")
	}
	if !resp.WalHealthy() {
		t.Error("wal_healthy = false, want true for an engine with no log")
	}
}

func TestStatsReportsWALState(t *testing.T) {
	dir := t.TempDir()
	engine, _, err := core.Open(core.Options{
		Dims:     testDims,
		Capacity: 64,
		Snapshot: filepath.Join(dir, "snap.mindb"),
		WAL:      filepath.Join(dir, "snap.mindb.wal"),
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { engine.Close() })

	client := startServer(t, engine, "")
	resp, err := client.Stats(ctxWithTimeout(t), buildStats())
	if err != nil {
		t.Fatal(err)
	}
	if !resp.WalEnabled() {
		t.Error("wal_enabled = false, want true")
	}
	if !resp.WalHealthy() {
		t.Error("wal_healthy = false on a log that has not failed")
	}
}
