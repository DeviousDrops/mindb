package telemetry

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/DeviousDrops/mindb/pkg/core"
	"github.com/DeviousDrops/mindb/pkg/metrics"
)

// sampleLine is the shape of every non-comment line in the text format.
var sampleLine = regexp.MustCompile(`^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? (\S+)$`)

// parse checks the whole scrape is well formed and returns name{labels} -> value.
// It is the check Prometheus itself would apply: one bad line fails the scrape.
func parse(t *testing.T, text string) map[string]float64 {
	t.Helper()
	out := map[string]float64{}
	typed := map[string]string{}
	for i, line := range strings.Split(strings.TrimSuffix(text, "\n"), "\n") {
		if strings.HasPrefix(line, "# HELP ") {
			continue
		}
		if strings.HasPrefix(line, "# TYPE ") {
			f := strings.Fields(line)
			if _, dup := typed[f[2]]; dup {
				t.Fatalf("line %d: family %s declared twice", i+1, f[2])
			}
			typed[f[2]] = f[3]
			continue
		}
		m := sampleLine.FindStringSubmatch(line)
		if m == nil {
			t.Fatalf("line %d is not a valid sample: %q", i+1, line)
		}
		base := m[1]
		for _, suf := range []string{"_bucket", "_sum", "_count"} {
			if typ, ok := typed[strings.TrimSuffix(base, suf)]; ok && typ == "histogram" {
				base = strings.TrimSuffix(base, suf)
			}
		}
		if _, ok := typed[base]; !ok {
			t.Fatalf("line %d: sample %s has no TYPE line before it", i+1, m[1])
		}
		v, err := strconv.ParseFloat(strings.Replace(m[3], "+Inf", "Inf", 1), 64)
		if err != nil {
			t.Fatalf("line %d: value %q: %v", i+1, m[3], err)
		}
		out[m[1]+m[2]] = v
	}
	return out
}

func scrape(t *testing.T, reg *metrics.Registry) map[string]float64 {
	t.Helper()
	var buf bytes.Buffer
	reg.WriteTo(&buf)
	return parse(t, buf.String())
}

// The full path: an engine with a log, driven by concurrent writers, a search
// and a snapshot, then scraped. The numbers have to agree with each other, not
// just be present.
func TestEndToEnd(t *testing.T) {
	const dims = 16
	dir := t.TempDir()
	reg := metrics.NewRegistry()
	tel := New(reg)

	e, _, err := core.Open(core.Options{
		Dims:     dims,
		Capacity: 1024,
		Snapshot: filepath.Join(dir, "snap.mindb"),
		WAL:      filepath.Join(dir, "snap.mindb.wal"),
		Observer: tel,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	tel.WatchEngine(e, "test")
	devs, diskErr := tel.WatchDisk(dir)
	ioOK := tel.WatchProcess()

	const writers, each = 8, 25
	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(w)))
			for i := 0; i < each; i++ {
				v := make([]float32, dims)
				for j := range v {
					v[j] = r.Float32()*2 - 1
				}
				if err := e.Insert(fmt.Sprintf("%d-%d", w, i), v, nil); err != nil {
					t.Error(err)
				}
			}
		}(w)
	}
	wg.Wait()

	q := make([]float32, dims)
	q[0] = 1
	e.SetCascade(false)
	if _, err := e.Search(q, 5); err != nil {
		t.Fatal(err)
	}
	e.SetCascade(true)
	if _, err := e.Search(q, 5); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(filepath.Join(dir, "snap.mindb")); err != nil {
		t.Fatal(err)
	}

	m := scrape(t, reg)
	want := func(key string, check func(float64) bool) {
		t.Helper()
		v, ok := m[key]
		if !ok {
			t.Errorf("missing %s", key)
			return
		}
		if !check(v) {
			t.Errorf("%s = %v", key, v)
		}
	}
	eq := func(x float64) func(float64) bool { return func(v float64) bool { return v == x } }
	pos := func(v float64) bool { return v > 0 }

	n := float64(writers * each)
	want("mindb_vectors", eq(n))
	want("mindb_wal_records_total", eq(n))
	want("mindb_wal_batch_records_sum", eq(n))
	want("mindb_wal_healthy", eq(1))
	want("mindb_wal_bytes_total", pos)
	want(`mindb_wal_syncs_total{kind="rotation"}`, eq(1))
	want(`mindb_search_duration_seconds_count{path="scan"}`, eq(1))
	want(`mindb_search_duration_seconds_count{path="cascade"}`, func(v float64) bool {
		// One search, on whichever path the cascade took for this data.
		return v+m[`mindb_search_duration_seconds_count{path="fallback"}`] == 1
	})
	want("mindb_search_cascade_survivor_ratio_count", eq(1))
	want("mindb_snapshot_duration_seconds_count", eq(1))
	want("mindb_snapshot_size_bytes", pos)
	want("mindb_snapshot_last_success_timestamp_seconds", pos)
	want("mindb_wal_segments", eq(1))
	want("go_goroutines", pos)

	commits := m[`mindb_wal_syncs_total{kind="commit"}`]
	if commits < 1 || commits > n {
		t.Errorf("%v commit syncs for %v writes", commits, n)
	}
	// Histogram invariants: +Inf bucket equals _count.
	if m[`mindb_wal_fsync_duration_seconds_bucket{le="+Inf"}`] != m["mindb_wal_fsync_duration_seconds_count"] {
		t.Error("fsync histogram +Inf bucket disagrees with _count")
	}
	t.Logf("%v writes, %v commit fsyncs, mean batch %.1f", n, commits, n/commits)

	if diskErr != nil {
		t.Logf("no disk metrics here: %v", diskErr)
	} else {
		label := fmt.Sprintf(`{device="%s"}`, devs[0].Name)
		want("mindb_disk_written_bytes_total"+label, func(v float64) bool { return v >= 0 })
		want("mindb_disk_writes_completed_total"+label, func(v float64) bool { return v >= 0 })
	}
	if ioOK {
		want("mindb_process_io_write_chars_total", pos)
	}
}

func TestInterceptor(t *testing.T) {
	reg := metrics.NewRegistry()
	tel := New(reg)
	ic := tel.UnaryInterceptor()

	info := &grpc.UnaryServerInfo{FullMethod: "/mindb.VectorService/Search"}
	ok := func(context.Context, any) (any, error) { return "resp", nil }
	bad := func(context.Context, any) (any, error) {
		return nil, status.Error(codes.InvalidArgument, "no")
	}

	for i := 0; i < 3; i++ {
		if resp, err := ic(context.Background(), nil, info, ok); err != nil || resp != "resp" {
			t.Fatalf("interceptor changed the result: %v %v", resp, err)
		}
	}
	if _, err := ic(context.Background(), nil, info, bad); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("interceptor changed the error: %v", err)
	}

	m := scrape(t, reg)
	if v := m[`mindb_grpc_requests_total{method="Search",code="OK"}`]; v != 3 {
		t.Errorf("OK count %v", v)
	}
	if v := m[`mindb_grpc_requests_total{method="Search",code="InvalidArgument"}`]; v != 1 {
		t.Errorf("InvalidArgument count %v", v)
	}
	if v := m[`mindb_grpc_request_duration_seconds_count{method="Search"}`]; v != 4 {
		t.Errorf("duration count %v", v)
	}
}
