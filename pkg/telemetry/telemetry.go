// Package telemetry turns what the engine, the gRPC server, the block device
// and the kernel know into Prometheus metrics.
//
// There are two kinds. Event metrics are updated as things happen: the engine
// reports every log sync, search and snapshot through core.Observer, and the
// gRPC interceptor reports every request. Scrape metrics are read when
// Prometheus asks: engine Stats, log size on disk, /proc/diskstats for the data
// device and /proc/self/io for this process. Nothing here runs a background
// goroutine.
//
// The metric that matters most is the pair of WAL ones: fsync latency and
// group-commit batch size. Under load the first should stay flat while the
// second grows — that is group commit working. Fsync latency climbing while
// batch size stays at 1 is a disk that has become the bottleneck.
package telemetry

import (
	"context"
	"runtime"
	"runtime/metrics"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/status"

	"github.com/DeviousDrops/mindb/pkg/core"
	"github.com/DeviousDrops/mindb/pkg/diskstats"
	mmetrics "github.com/DeviousDrops/mindb/pkg/metrics"
)

// Telemetry is a core.Observer that records into a metrics registry.
type Telemetry struct {
	reg *mmetrics.Registry

	walFsync    *mmetrics.Histogram
	walWrite    *mmetrics.Histogram
	walBatch    *mmetrics.Histogram
	walSyncs    *mmetrics.CounterVec
	walRecords  *mmetrics.Counter
	walBytes    *mmetrics.Counter
	walSyncErrs *mmetrics.Counter
	writeApply  *mmetrics.Histogram

	search    *mmetrics.HistogramVec
	survivors *mmetrics.Histogram

	snapDuration *mmetrics.Histogram
	snapBytes    *mmetrics.Gauge
	snapLastOK   *mmetrics.Gauge
	snapErrs     *mmetrics.Counter

	grpcRequests *mmetrics.CounterVec
	grpcDuration *mmetrics.HistogramVec
}

var _ core.Observer = (*Telemetry)(nil)

// Latency buckets, chosen for what each path costs on real hardware.
var (
	// fsync: ~50 µs on a laptop NVMe with a write cache, ~1-10 ms on a cloud
	// volume, seconds on a device that is in trouble.
	fsyncBuckets = mmetrics.ExponentialBuckets(50e-6, 2, 16) // 50 µs .. 1.6 s

	// write(): a memcpy into the page cache. Microseconds unless the kernel
	// is throttling dirty pages, which is exactly when it is worth seeing.
	writeBuckets = mmetrics.ExponentialBuckets(1e-6, 4, 11) // 1 µs .. 1 s

	// Batch sizes are powers of two up to well past any realistic writer count.
	batchBuckets = mmetrics.ExponentialBuckets(1, 2, 12) // 1 .. 2048

	// Search: sub-millisecond on a small corpus, tens of ms on 100k x 768
	// without SIMD.
	searchBuckets = mmetrics.ExponentialBuckets(50e-6, 2, 16) // 50 µs .. 1.6 s

	// Survivor ratio: clustered data measures ~0.4% and the guard fires
	// above 25%, so the interesting range is three orders of magnitude.
	survivorBuckets = []float64{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1}

	snapshotBuckets = mmetrics.ExponentialBuckets(1e-3, 2, 16) // 1 ms .. 33 s
	rpcBuckets      = mmetrics.ExponentialBuckets(50e-6, 2, 16)
)

// New registers the event metrics on reg and returns the observer to pass as
// core.Options.Observer.
func New(reg *mmetrics.Registry) *Telemetry {
	return &Telemetry{
		reg: reg,

		walFsync: reg.Histogram("mindb_wal_fsync_duration_seconds",
			"Time for fsync of the write-ahead log, per group commit.", fsyncBuckets),
		walWrite: reg.Histogram("mindb_wal_write_duration_seconds",
			"Time to write buffered log records into the page cache, per group commit.", writeBuckets),
		walBatch: reg.Histogram("mindb_wal_batch_records",
			"Records made durable by one fsync: the group-commit batch size.", batchBuckets),
		walSyncs: reg.CounterVec("mindb_wal_syncs_total",
			"Log syncs, by kind: commit (a writer waited on it) or rotation (before a snapshot).", "kind"),
		walRecords: reg.Counter("mindb_wal_records_total",
			"Records made durable in the write-ahead log."),
		walBytes: reg.Counter("mindb_wal_bytes_total",
			"Bytes of log records made durable. The denominator of write amplification."),
		walSyncErrs: reg.Counter("mindb_wal_sync_errors_total",
			"Log writes or fsyncs that failed. Any increase means the log is now unhealthy for good."),
		writeApply: reg.Histogram("mindb_write_apply_duration_seconds",
			"Time for an Insert or Delete to get through the write lock and into the slab, before logging. Almost all lock wait: behind other writers, then behind running searches.",
			searchBuckets),

		search: reg.HistogramVec("mindb_search_duration_seconds",
			"Search latency including the wait for the read lock, by path: scan, cascade, or fallback.",
			searchBuckets, "path"),
		survivors: reg.Histogram("mindb_search_cascade_survivor_ratio",
			"Fraction of live vectors that survived the cascade's int8 pruning and were rescored exactly.",
			survivorBuckets),

		snapDuration: reg.Histogram("mindb_snapshot_duration_seconds",
			"Time for a whole Save: log rotation, write, fsync, rename, meta.", snapshotBuckets),
		snapBytes: reg.Gauge("mindb_snapshot_size_bytes",
			"Size of the last snapshot written successfully."),
		snapLastOK: reg.Gauge("mindb_snapshot_last_success_timestamp_seconds",
			"Unix time of the last successful snapshot. The log grows until the next one."),
		snapErrs: reg.Counter("mindb_snapshot_errors_total",
			"Snapshots that failed."),

		grpcRequests: reg.CounterVec("mindb_grpc_requests_total",
			"gRPC requests handled, by method and status code.", "method", "code"),
		grpcDuration: reg.HistogramVec("mindb_grpc_request_duration_seconds",
			"gRPC handler latency by method, excluding network and codec time.", rpcBuckets, "method"),
	}
}

// WALSync implements core.Observer.
func (t *Telemetry) WALSync(e core.WALSync) {
	kind := "commit"
	if e.Rotation {
		kind = "rotation"
	}
	t.walSyncs.With(kind).Inc()
	if e.Err != nil {
		t.walSyncErrs.Inc()
		return
	}
	t.walWrite.Observe(e.Write.Seconds())
	t.walFsync.Observe(e.Fsync.Seconds())
	if e.Records > 0 {
		// A rotation with nothing pending is a sync of an empty buffer; it says
		// nothing about batching and would drag the histogram toward zero.
		t.walBatch.Observe(float64(e.Records))
		t.walRecords.Add(uint64(e.Records))
		t.walBytes.Add(uint64(e.Bytes))
	}
}

// WriteApply implements core.Observer.
func (t *Telemetry) WriteApply(d time.Duration) { t.writeApply.Observe(d.Seconds()) }

// Search implements core.Observer.
func (t *Telemetry) Search(e core.SearchEvent) {
	t.search.With(e.Path.String()).Observe(e.Duration.Seconds())
	if e.Path != core.PathScan && e.Live > 0 {
		t.survivors.Observe(float64(e.Survivors) / float64(e.Live))
	}
}

// Snapshot implements core.Observer.
func (t *Telemetry) Snapshot(e core.SnapshotEvent) {
	t.snapDuration.Observe(e.Duration.Seconds())
	if e.Err != nil {
		t.snapErrs.Inc()
		return
	}
	t.snapBytes.Set(float64(e.Bytes))
	t.snapLastOK.Set(float64(time.Now().UnixNano()) / 1e9)
}

// UnaryInterceptor records every unary RPC's latency and status code.
//
// The duration is the handler's: it starts after the request is decoded and
// stops before the response is encoded, so it isolates the engine from the
// network. Compare it with a client-side measurement to see what the wire costs.
func (t *Telemetry) UnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		method := info.FullMethod[strings.LastIndexByte(info.FullMethod, '/')+1:]
		t.grpcDuration.With(method).Observe(time.Since(start).Seconds())
		t.grpcRequests.With(method, status.Code(err).String()).Inc()
		return resp, err
	}
}

// WatchEngine registers gauges read from the engine at scrape time.
func (t *Telemetry) WatchEngine(e *core.Engine, version string) {
	t.reg.Register(engineCollector{e: e, version: version})
}

type engineCollector struct {
	e       *core.Engine
	version string
}

func (c engineCollector) Collect(enc *mmetrics.Encoder) {
	s := c.e.Stats()

	enc.Family("mindb_build_info", "Build and runtime facts; always 1.", "gauge")
	enc.Sample("mindb_build_info", []mmetrics.Label{
		{Name: "version", Value: c.version},
		{Name: "goversion", Value: runtime.Version()},
		{Name: "goarch", Value: s.GoArch},
		{Name: "kernel", Value: s.KernelName},
		{Name: "fast_int8", Value: strconv.FormatBool(s.FastInt8)},
	}, 1)

	gauge(enc, "mindb_vectors", "Live vectors.", float64(s.Count))
	gauge(enc, "mindb_capacity_vectors", "Slots reserved at boot.", float64(s.Capacity))
	gauge(enc, "mindb_memory_reserved_bytes", "Memory reserved for the vector slab, allocated eagerly at boot.", float64(s.MemoryBytes))
	gauge(enc, "mindb_payload_bytes", "Bytes of payload held for live vectors.", float64(s.PayloadBytes))
	gauge(enc, "mindb_wal_enabled", "1 if writes are logged.", b2f(s.WALEnabled))
	gauge(enc, "mindb_wal_healthy", "1 while every log write has reached disk. Goes to 0 on the first failure and stays there.", b2f(s.WALHealthy))

	if s.WALEnabled {
		if segs, bytes, err := c.e.WALDiskUsage(); err == nil {
			gauge(enc, "mindb_wal_segments", "Log segments on disk. More than one between snapshots means a rotation is pending deletion.", float64(segs))
			gauge(enc, "mindb_wal_disk_bytes", "Bytes of log on disk. Only a snapshot shrinks it.", float64(bytes))
		}
	}
}

// WatchDisk registers /proc/diskstats counters for the block device holding
// dir, and for its whole disk if that device is a partition. It returns the
// devices found, or the reason there are none — which is routine inside a
// container whose root is overlayfs, so the caller should log it, not fail.
func (t *Telemetry) WatchDisk(dir string) ([]diskstats.Device, error) {
	devs, err := diskstats.DeviceFor(dir)
	if err != nil {
		return nil, err
	}
	t.reg.Register(diskCollector{devs: devs})
	return devs, nil
}

type diskCollector struct{ devs []diskstats.Device }

func (c diskCollector) Collect(enc *mmetrics.Encoder) {
	type reading struct {
		labels []mmetrics.Label
		s      diskstats.Stats
	}
	var rs []reading
	for _, d := range c.devs {
		s, err := diskstats.Read(d)
		if err != nil {
			continue // a device that vanished mid-run; nothing useful to say
		}
		rs = append(rs, reading{[]mmetrics.Label{{Name: "device", Value: d.Name}}, s})
	}
	if len(rs) == 0 {
		return
	}

	const ms = 1e-3
	family := func(name, help, typ string, v func(s diskstats.Stats) (float64, bool)) {
		header := false
		for _, r := range rs {
			x, ok := v(r.s)
			if !ok {
				continue
			}
			if !header {
				enc.Family(name, help, typ)
				header = true
			}
			enc.Sample(name, r.labels, x)
		}
	}
	always := func(f func(s diskstats.Stats) float64) func(diskstats.Stats) (float64, bool) {
		return func(s diskstats.Stats) (float64, bool) { return f(s), true }
	}

	family("mindb_disk_reads_completed_total", "Reads completed by the data device.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.ReadsCompleted) }))
	family("mindb_disk_writes_completed_total", "Writes completed by the data device.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.WritesCompleted) }))
	family("mindb_disk_writes_merged_total", "Adjacent writes the I/O scheduler merged before issuing.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.WritesMerged) }))
	family("mindb_disk_read_bytes_total", "Bytes read from the data device.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.SectorsRead * diskstats.SectorSize) }))
	family("mindb_disk_written_bytes_total", "Bytes written to the data device, by every process on it.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.SectorsWritten * diskstats.SectorSize) }))
	family("mindb_disk_read_time_seconds_total", "Time reads spent queued and in service. Divide its rate by the read rate for mean read latency.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.ReadTimeMs) * ms }))
	family("mindb_disk_write_time_seconds_total", "Time writes spent queued and in service. Divide its rate by the write rate for mean write latency.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.WriteTimeMs) * ms }))
	family("mindb_disk_io_in_progress", "Requests issued to the device and not yet completed.", "gauge",
		always(func(s diskstats.Stats) float64 { return float64(s.InFlight) }))
	family("mindb_disk_io_time_seconds_total", "Time the device had at least one request in flight. Its rate is utilization.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.IOTimeMs) * ms }))
	family("mindb_disk_io_weighted_time_seconds_total", "In-flight time weighted by queue depth. Its rate is the mean queue depth.", "counter",
		always(func(s diskstats.Stats) float64 { return float64(s.WeightedIOTimeMs) * ms }))
	family("mindb_disk_flushes_total", "Cache flush requests completed: the device side of an fsync. Kernel 5.5+.", "counter",
		func(s diskstats.Stats) (float64, bool) { return float64(s.FlushesCompleted), s.HasFlush })
	family("mindb_disk_flush_time_seconds_total", "Time spent on cache flushes. Kernel 5.5+.", "counter",
		func(s diskstats.Stats) (float64, bool) { return float64(s.FlushTimeMs) * ms, s.HasFlush })
}

// WatchProcess registers this process's I/O counters from /proc/self/io and a
// few Go runtime gauges. It reports whether per-process I/O accounting was
// available; the runtime gauges are registered either way.
func (t *Telemetry) WatchProcess() bool {
	_, err := diskstats.ReadSelfIO()
	ioOK := err == nil
	t.reg.Register(processCollector{io: ioOK, start: time.Now()})
	return ioOK
}

type processCollector struct {
	io    bool
	start time.Time
}

var runtimeSamples = []metrics.Sample{
	{Name: "/sched/goroutines:goroutines"},
	{Name: "/memory/classes/heap/objects:bytes"},
	{Name: "/gc/cycles/total:gc-cycles"},
}

func (c processCollector) Collect(enc *mmetrics.Encoder) {
	gauge(enc, "process_start_time_seconds", "Unix time the process started.", float64(c.start.UnixNano())/1e9)

	samples := make([]metrics.Sample, len(runtimeSamples))
	copy(samples, runtimeSamples)
	metrics.Read(samples)
	gauge(enc, "go_goroutines", "Goroutines that currently exist.", float64(samples[0].Value.Uint64()))
	gauge(enc, "go_heap_objects_bytes", "Bytes in live and not-yet-swept heap objects.", float64(samples[1].Value.Uint64()))
	enc.Family("go_gc_cycles_total", "Completed GC cycles.", "counter")
	enc.Sample("go_gc_cycles_total", nil, float64(samples[2].Value.Uint64()))

	if !c.io {
		return
	}
	p, err := diskstats.ReadSelfIO()
	if err != nil {
		return
	}
	counter := func(name, help string, v uint64) {
		enc.Family(name, help, "counter")
		enc.Sample(name, nil, float64(v))
	}
	counter("mindb_process_io_write_bytes_total",
		"Bytes this process caused to be written to storage, charged when a page is first dirtied. Divide its rate by mindb_wal_bytes_total's for write amplification.", p.WriteBytes)
	counter("mindb_process_io_read_bytes_total",
		"Bytes this process caused to be read from storage.", p.ReadBytes)
	counter("mindb_process_io_cancelled_write_bytes_total",
		"Dirty bytes never written because their file was truncated or deleted first.", p.CancelledWriteBytes)
	counter("mindb_process_io_write_chars_total",
		"Bytes passed to write() and friends, whether or not they reached storage.", p.WChar)
	counter("mindb_process_io_write_syscalls_total",
		"write() and friends: syscall count.", p.SyscW)
}

func gauge(enc *mmetrics.Encoder, name, help string, v float64) {
	enc.Family(name, help, "gauge")
	enc.Sample(name, nil, v)
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
