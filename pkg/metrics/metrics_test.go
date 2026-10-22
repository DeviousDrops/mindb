package metrics

import (
	"bytes"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func scrape(r *Registry) string {
	var buf bytes.Buffer
	r.WriteTo(&buf)
	return buf.String()
}

// The exact text matters: Prometheus rejects a whole scrape over one bad line,
// so this pins the format byte for byte rather than checking for substrings.
func TestExpositionFormat(t *testing.T) {
	r := NewRegistry()
	c := r.Counter("test_events_total", "Events seen.")
	g := r.Gauge("test_temperature", "Current temperature.")
	h := r.Histogram("test_latency_seconds", "Latency.", []float64{0.1, 1})
	cv := r.CounterVec("test_requests_total", "Requests.", "method", "code")

	c.Add(3)
	g.Set(-1.5)
	h.Observe(0.05)
	h.Observe(0.1) // on the boundary: buckets are "less than or equal"
	h.Observe(5)
	cv.With("Search", "OK").Inc()
	cv.With("Insert", "OK").Add(2)

	want := `# HELP test_events_total Events seen.
# TYPE test_events_total counter
test_events_total 3
# HELP test_temperature Current temperature.
# TYPE test_temperature gauge
test_temperature -1.5
# HELP test_latency_seconds Latency.
# TYPE test_latency_seconds histogram
test_latency_seconds_bucket{le="0.1"} 2
test_latency_seconds_bucket{le="1"} 2
test_latency_seconds_bucket{le="+Inf"} 3
test_latency_seconds_sum 5.15
test_latency_seconds_count 3
# HELP test_requests_total Requests.
# TYPE test_requests_total counter
test_requests_total{method="Insert",code="OK"} 2
test_requests_total{method="Search",code="OK"} 1
`
	if got := scrape(r); got != want {
		t.Fatalf("exposition mismatch\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func TestEscaping(t *testing.T) {
	r := NewRegistry()
	r.CounterVec("test_total", "Back\\slash and\nnewline.", "path").With("C:\\data\n\"x\"").Inc()

	got := scrape(r)
	for _, want := range []string{
		`# HELP test_total Back\\slash and\nnewline.`,
		`test_total{path="C:\\data\n\"x\""} 1`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in\n%s", want, got)
		}
	}
}

func TestSpecialFloats(t *testing.T) {
	for v, want := range map[float64]string{
		math.Inf(+1): "+Inf",
		math.Inf(-1): "-Inf",
		0.25:         "0.25",
		1e-6:         "1e-06",
	} {
		if got := formatFloat(v); got != want {
			t.Errorf("formatFloat(%v) = %q, want %q", v, got, want)
		}
	}
	if got := formatFloat(math.NaN()); got != "NaN" {
		t.Errorf("formatFloat(NaN) = %q", got)
	}
}

// Every Observe from many goroutines must land: the +Inf bucket, _count and
// the sum all have to agree once the writers are done. Run under -race in CI.
func TestHistogramConcurrent(t *testing.T) {
	h := newHistogram(ExponentialBuckets(1, 2, 8))
	const workers, each = 16, 5000

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				h.Observe(float64(i%300) + 0.5)
			}
		}(w)
	}
	wg.Wait()

	if got := h.Count(); got != workers*each {
		t.Fatalf("count = %d, want %d", got, workers*each)
	}
	var wantSum float64
	for i := 0; i < each; i++ {
		wantSum += float64(i%300) + 0.5
	}
	wantSum *= workers
	if got := h.Sum(); math.Abs(got-wantSum) > 1e-6*wantSum {
		t.Fatalf("sum = %v, want %v", got, wantSum)
	}
}

func TestBucketsAreCumulative(t *testing.T) {
	r := NewRegistry()
	h := r.Histogram("b", "b", []float64{1, 2, 4})
	for _, v := range []float64{0, 1, 1.5, 3, 3, 100} {
		h.Observe(v)
	}
	got := scrape(r)
	for _, want := range []string{
		`b_bucket{le="1"} 2`,
		`b_bucket{le="2"} 3`,
		`b_bucket{le="4"} 5`,
		`b_bucket{le="+Inf"} 6`,
		`b_count 6`,
	} {
		if !strings.Contains(got, want+"\n") {
			t.Errorf("missing %q in\n%s", want, got)
		}
	}
}

func TestDuplicateNamePanics(t *testing.T) {
	r := NewRegistry()
	r.Counter("dup_total", "x")
	defer func() {
		if recover() == nil {
			t.Fatal("registering a name twice should panic")
		}
	}()
	r.Gauge("dup_total", "x")
}

func TestInvalidNamesPanic(t *testing.T) {
	for name, register := range map[string]func(r *Registry){
		"metric name": func(r *Registry) { r.Counter("9bad", "x") },
		"label le":    func(r *Registry) { r.CounterVec("ok_total", "x", "le") },
		"label dunder": func(r *Registry) {
			r.CounterVec("ok_total", "x", "__name")
		},
		"buckets unsorted": func(r *Registry) { r.Histogram("h", "x", []float64{2, 1}) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("want panic")
				}
			}()
			register(NewRegistry())
		})
	}
}

func TestGaugeFuncReadsAtScrape(t *testing.T) {
	r := NewRegistry()
	v := 1.0
	r.GaugeFunc("live", "x", func() float64 { return v })
	if !strings.Contains(scrape(r), "live 1\n") {
		t.Fatal("first scrape")
	}
	v = 7
	if !strings.Contains(scrape(r), "live 7\n") {
		t.Fatal("value should be read at scrape time, not registration")
	}
}

func TestHandler(t *testing.T) {
	r := NewRegistry()
	r.Counter("hits_total", "x").Inc()

	rec := httptest.NewRecorder()
	r.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != ContentType {
		t.Fatalf("content type %q", ct)
	}
	if !strings.Contains(rec.Body.String(), "hits_total 1\n") {
		t.Fatalf("body:\n%s", rec.Body.String())
	}

	rec = httptest.NewRecorder()
	r.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/metrics", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status %d", rec.Code)
	}
}

func BenchmarkHistogramObserve(b *testing.B) {
	h := newHistogram(ExponentialBuckets(50e-6, 2, 16))
	b.RunParallel(func(pb *testing.PB) {
		v := 0.0003
		for pb.Next() {
			h.Observe(v)
		}
	})
}
