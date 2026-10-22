// Package metrics is a small Prometheus instrumentation library: counters,
// gauges and histograms, written out in the Prometheus text exposition format
// (version 0.0.4).
//
// It exists instead of prometheus/client_golang. The official client is the
// right choice for a large service, but it brings roughly ten modules with it,
// and MinDB's whole dependency list is FlatBuffers, gRPC and x/sys. What MinDB
// needs from it is small: a handful of metric types and a text encoder, which is
// what this is. See docs/DECISIONS.md.
//
// Every update is a single atomic operation, so the hot paths that record
// metrics (the WAL group commit and Search) never take a lock to do it.
package metrics

import (
	"bytes"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// Collector writes one or more metric families when the registry is scraped.
//
// Everything in this package is a Collector, and so is anything that reads its
// values at scrape time rather than being updated as events happen — the disk
// statistics, for instance, are a file read once per scrape.
type Collector interface {
	Collect(e *Encoder)
}

// Registry is an ordered set of collectors, served at /metrics.
type Registry struct {
	mu         sync.Mutex
	collectors []Collector
	names      map[string]bool
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{names: make(map[string]bool)}
}

var validName = regexp.MustCompile(`^[a-zA-Z_:][a-zA-Z0-9_:]*$`)

// claim reserves a family name, panicking on a duplicate or an invalid name.
// Both are programming errors that would otherwise surface as a scrape that
// Prometheus rejects outright, which is a worse place to find them.
func (r *Registry) claim(name string) {
	if !validName.MatchString(name) {
		panic(fmt.Sprintf("metrics: invalid metric name %q", name))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.names[name] {
		panic(fmt.Sprintf("metrics: %q registered twice", name))
	}
	r.names[name] = true
}

// Register adds a collector. Families it emits are not checked for duplicates
// against the rest of the registry; the typed constructors below are.
func (r *Registry) Register(c Collector) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.collectors = append(r.collectors, c)
}

// WriteTo encodes every registered collector in registration order.
func (r *Registry) WriteTo(buf *bytes.Buffer) {
	r.mu.Lock()
	cs := append([]Collector(nil), r.collectors...)
	r.mu.Unlock()

	e := &Encoder{buf: buf}
	for _, c := range cs {
		c.Collect(e)
	}
}

// ContentType is what Prometheus expects for the text format.
const ContentType = "text/plain; version=0.0.4; charset=utf-8"

// Handler serves the registry. Each scrape encodes into a fresh buffer, so a
// slow client never holds anything the instrumented code is waiting on.
func (r *Registry) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet && req.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var buf bytes.Buffer
		r.WriteTo(&buf)
		w.Header().Set("Content-Type", ContentType)
		w.Header().Set("Content-Length", strconv.Itoa(buf.Len()))
		if req.Method == http.MethodGet {
			w.Write(buf.Bytes())
		}
	})
}

// ---------------------------------------------------------------------------
// Encoding

// Label is one name="value" pair on a sample.
type Label struct{ Name, Value string }

// Encoder writes families and samples in the text format.
type Encoder struct {
	buf *bytes.Buffer
}

// Family writes the HELP and TYPE lines that precede a family's samples.
// typ is "counter", "gauge", "histogram" or "untyped".
func (e *Encoder) Family(name, help, typ string) {
	e.buf.WriteString("# HELP ")
	e.buf.WriteString(name)
	e.buf.WriteByte(' ')
	e.buf.WriteString(escapeHelp(help))
	e.buf.WriteString("\n# TYPE ")
	e.buf.WriteString(name)
	e.buf.WriteByte(' ')
	e.buf.WriteString(typ)
	e.buf.WriteByte('\n')
}

// Sample writes one line: name{labels} value.
func (e *Encoder) Sample(name string, labels []Label, v float64) {
	e.buf.WriteString(name)
	writeLabels(e.buf, labels)
	e.buf.WriteByte(' ')
	e.buf.WriteString(formatFloat(v))
	e.buf.WriteByte('\n')
}

// sampleUint is Sample for a counter, which keeps full integer precision past
// 2^53 rather than going through a float64.
func (e *Encoder) sampleUint(name string, labels []Label, v uint64) {
	e.buf.WriteString(name)
	writeLabels(e.buf, labels)
	e.buf.WriteByte(' ')
	e.buf.WriteString(strconv.FormatUint(v, 10))
	e.buf.WriteByte('\n')
}

func writeLabels(buf *bytes.Buffer, labels []Label) {
	if len(labels) == 0 {
		return
	}
	buf.WriteByte('{')
	for i, l := range labels {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(l.Name)
		buf.WriteString(`="`)
		buf.WriteString(escapeLabel(l.Value))
		buf.WriteByte('"')
	}
	buf.WriteByte('}')
}

var (
	helpEscaper  = strings.NewReplacer(`\`, `\\`, "\n", `\n`)
	labelEscaper = strings.NewReplacer(`\`, `\\`, "\n", `\n`, `"`, `\"`)
)

func escapeHelp(s string) string  { return helpEscaper.Replace(s) }
func escapeLabel(s string) string { return labelEscaper.Replace(s) }

func formatFloat(v float64) string {
	switch {
	case math.IsInf(v, +1):
		return "+Inf"
	case math.IsInf(v, -1):
		return "-Inf"
	case math.IsNaN(v):
		return "NaN"
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// ---------------------------------------------------------------------------
// Counter

// Counter is a monotonically increasing integer.
type Counter struct{ v atomic.Uint64 }

// Inc adds one.
func (c *Counter) Inc() { c.v.Add(1) }

// Add adds n.
func (c *Counter) Add(n uint64) { c.v.Add(n) }

// Value returns the current count.
func (c *Counter) Value() uint64 { return c.v.Load() }

type counterFamily struct {
	name, help string
	c          *Counter
}

func (f *counterFamily) Collect(e *Encoder) {
	e.Family(f.name, f.help, "counter")
	e.sampleUint(f.name, nil, f.c.Value())
}

// Counter registers and returns an unlabelled counter.
func (r *Registry) Counter(name, help string) *Counter {
	r.claim(name)
	c := &Counter{}
	r.Register(&counterFamily{name: name, help: help, c: c})
	return c
}

// ---------------------------------------------------------------------------
// Gauge

// Gauge is a float64 that can go up and down.
type Gauge struct{ bits atomic.Uint64 }

// Set replaces the value.
func (g *Gauge) Set(v float64) { g.bits.Store(math.Float64bits(v)) }

// Value returns the current value.
func (g *Gauge) Value() float64 { return math.Float64frombits(g.bits.Load()) }

type gaugeFamily struct {
	name, help string
	g          *Gauge
}

func (f *gaugeFamily) Collect(e *Encoder) {
	e.Family(f.name, f.help, "gauge")
	e.Sample(f.name, nil, f.g.Value())
}

// Gauge registers and returns an unlabelled gauge.
func (r *Registry) Gauge(name, help string) *Gauge {
	r.claim(name)
	g := &Gauge{}
	r.Register(&gaugeFamily{name: name, help: help, g: g})
	return g
}

type funcFamily struct {
	name, help, typ string
	fn              func() float64
}

func (f *funcFamily) Collect(e *Encoder) {
	e.Family(f.name, f.help, f.typ)
	e.Sample(f.name, nil, f.fn())
}

// GaugeFunc registers a gauge whose value is read by calling fn at scrape time.
// fn runs on the scrape goroutine and must not block for long.
func (r *Registry) GaugeFunc(name, help string, fn func() float64) {
	r.claim(name)
	r.Register(&funcFamily{name: name, help: help, typ: "gauge", fn: fn})
}

// ---------------------------------------------------------------------------
// Histogram

// Histogram counts observations into fixed buckets.
//
// Bucket counts are stored per bucket and made cumulative only when written,
// so Observe is one binary search and two atomic adds. _count is derived from
// the bucket counts at scrape time rather than kept separately, which keeps it
// exactly equal to the +Inf bucket as the format requires — a separate counter
// read a moment later could disagree under concurrent Observes. _sum can still
// trail the buckets by an in-flight observation; client_golang removes that with
// a hot/cold double buffer, which is more machinery than a dashboard needs.
type Histogram struct {
	upper  []float64       // bucket upper bounds, ascending, without +Inf
	counts []atomic.Uint64 // len(upper)+1; the last is the +Inf bucket
	sum    atomic.Uint64   // float64 bits
}

func newHistogram(buckets []float64) *Histogram {
	if len(buckets) == 0 {
		panic("metrics: histogram needs at least one bucket")
	}
	if !sort.Float64sAreSorted(buckets) {
		panic("metrics: histogram buckets must be ascending")
	}
	for i := 1; i < len(buckets); i++ {
		if buckets[i] == buckets[i-1] {
			panic("metrics: duplicate histogram bucket")
		}
	}
	return &Histogram{
		upper:  append([]float64(nil), buckets...),
		counts: make([]atomic.Uint64, len(buckets)+1),
	}
}

// Observe records one value.
func (h *Histogram) Observe(v float64) {
	// First bucket whose upper bound is >= v; buckets are "less than or equal".
	i := sort.SearchFloat64s(h.upper, v)
	h.counts[i].Add(1)
	for {
		old := h.sum.Load()
		if h.sum.CompareAndSwap(old, math.Float64bits(math.Float64frombits(old)+v)) {
			return
		}
	}
}

// Count returns the number of observations so far.
func (h *Histogram) Count() uint64 {
	var n uint64
	for i := range h.counts {
		n += h.counts[i].Load()
	}
	return n
}

// Sum returns the sum of every observed value.
func (h *Histogram) Sum() float64 { return math.Float64frombits(h.sum.Load()) }

func (h *Histogram) write(e *Encoder, name string, labels []Label) {
	le := make([]Label, len(labels)+1)
	copy(le, labels)
	var cum uint64
	for i := range h.counts {
		cum += h.counts[i].Load()
		bound := math.Inf(+1)
		if i < len(h.upper) {
			bound = h.upper[i]
		}
		le[len(labels)] = Label{"le", formatFloat(bound)}
		e.sampleUint(name+"_bucket", le, cum)
	}
	e.Sample(name+"_sum", labels, h.Sum())
	e.sampleUint(name+"_count", labels, cum)
}

type histogramFamily struct {
	name, help string
	h          *Histogram
}

func (f *histogramFamily) Collect(e *Encoder) {
	e.Family(f.name, f.help, "histogram")
	f.h.write(e, f.name, nil)
}

// Histogram registers and returns an unlabelled histogram.
func (r *Registry) Histogram(name, help string, buckets []float64) *Histogram {
	r.claim(name)
	h := newHistogram(buckets)
	r.Register(&histogramFamily{name: name, help: help, h: h})
	return h
}

// ExponentialBuckets returns n upper bounds: start, start*factor, ...
func ExponentialBuckets(start, factor float64, n int) []float64 {
	if start <= 0 || factor <= 1 || n < 1 {
		panic("metrics: ExponentialBuckets needs start > 0, factor > 1, n >= 1")
	}
	b := make([]float64, n)
	for i := range b {
		b[i] = start
		start *= factor
	}
	return b
}

// ---------------------------------------------------------------------------
// Labelled families

// vec holds one series per distinct combination of label values. Lookups go
// through a sync.Map, so the steady state — every series already exists — is
// lock-free; only the first observation of a new combination takes the mutex.
type vec[T any] struct {
	name, help, typ string
	labelNames      []string
	newT            func() T
	write           func(e *Encoder, name string, labels []Label, t T)

	series sync.Map // key string -> *seriesOf[T]
	mu     sync.Mutex
}

type seriesOf[T any] struct {
	labels []Label
	t      T
}

func (v *vec[T]) with(values []string) T {
	if len(values) != len(v.labelNames) {
		panic(fmt.Sprintf("metrics: %s takes %d label values, got %d", v.name, len(v.labelNames), len(values)))
	}
	key := strings.Join(values, "\xff")
	if s, ok := v.series.Load(key); ok {
		return s.(*seriesOf[T]).t
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	if s, ok := v.series.Load(key); ok {
		return s.(*seriesOf[T]).t
	}
	labels := make([]Label, len(values))
	for i, val := range values {
		labels[i] = Label{v.labelNames[i], val}
	}
	s := &seriesOf[T]{labels: labels, t: v.newT()}
	v.series.Store(key, s)
	return s.t
}

func (v *vec[T]) Collect(e *Encoder) {
	var all []*seriesOf[T]
	v.series.Range(func(_, s any) bool {
		all = append(all, s.(*seriesOf[T]))
		return true
	})
	if len(all) == 0 {
		return // a family with no samples is valid but pointless to send
	}
	// Map iteration order is random; sorting keeps successive scrapes diffable.
	sort.Slice(all, func(i, j int) bool { return labelsLess(all[i].labels, all[j].labels) })

	e.Family(v.name, v.help, v.typ)
	for _, s := range all {
		v.write(e, v.name, s.labels, s.t)
	}
}

func labelsLess(a, b []Label) bool {
	for i := range a {
		if a[i].Value != b[i].Value {
			return a[i].Value < b[i].Value
		}
	}
	return false
}

func checkLabelNames(names []string) {
	for _, n := range names {
		if !validName.MatchString(n) || strings.Contains(n, ":") || strings.HasPrefix(n, "__") || n == "le" {
			panic(fmt.Sprintf("metrics: invalid label name %q", n))
		}
	}
}

// CounterVec is a counter with labels.
type CounterVec struct{ v *vec[*Counter] }

// With returns the counter for these label values, creating it on first use.
func (c *CounterVec) With(values ...string) *Counter { return c.v.with(values) }

// CounterVec registers a labelled counter family.
func (r *Registry) CounterVec(name, help string, labelNames ...string) *CounterVec {
	r.claim(name)
	checkLabelNames(labelNames)
	v := &vec[*Counter]{
		name: name, help: help, typ: "counter", labelNames: labelNames,
		newT: func() *Counter { return &Counter{} },
		write: func(e *Encoder, name string, labels []Label, c *Counter) {
			e.sampleUint(name, labels, c.Value())
		},
	}
	r.Register(v)
	return &CounterVec{v}
}

// HistogramVec is a histogram with labels; every series shares the buckets.
type HistogramVec struct{ v *vec[*Histogram] }

// With returns the histogram for these label values, creating it on first use.
func (h *HistogramVec) With(values ...string) *Histogram { return h.v.with(values) }

// HistogramVec registers a labelled histogram family.
func (r *Registry) HistogramVec(name, help string, buckets []float64, labelNames ...string) *HistogramVec {
	r.claim(name)
	checkLabelNames(labelNames)
	newHistogram(buckets) // validate now rather than on first observation
	v := &vec[*Histogram]{
		name: name, help: help, typ: "histogram", labelNames: labelNames,
		newT: func() *Histogram { return newHistogram(buckets) },
		write: func(e *Encoder, name string, labels []Label, h *Histogram) {
			h.write(e, name, labels)
		},
	}
	r.Register(v)
	return &HistogramVec{v}
}
