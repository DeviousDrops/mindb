// Command mindb-server runs MinDB as a gRPC sidecar.
package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	flatbuffers "github.com/google/flatbuffers/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/typicallhavok/mindb/pkg/api"
	"github.com/typicallhavok/mindb/pkg/core"
	"github.com/typicallhavok/mindb/pkg/math"
	"github.com/typicallhavok/mindb/pkg/mindb"
)

// serviceName is what the health service reports status for, alongside the
// empty string that means "the whole server".
const serviceName = "mindb.VectorService"

// version is stamped in at link time (-X main.version). It stays "dev" for a
// plain `go build`, which is the honest answer: an unstamped binary came from
// somebody's working tree and its git state is unknown.
var version = "dev"

// config is what the flags parse into.
type config struct {
	addr       string
	healthAddr string
	dims       int
	capacity   int
	snapPath   string
	snapWait   time.Duration
	wal        string
}

func main() {
	var cfg config
	flag.StringVar(&cfg.addr, "addr", ":50051", "gRPC listen address")
	flag.StringVar(&cfg.healthAddr, "health-addr", ":50052", "gRPC health service address; empty disables it")
	flag.IntVar(&cfg.dims, "dims", 768, "vector dimension (ignored when a snapshot is loaded)")
	flag.IntVar(&cfg.capacity, "capacity", 100_000, "maximum number of vectors; memory is reserved eagerly at boot")
	flag.StringVar(&cfg.snapPath, "snapshot", "", "snapshot file path; empty disables persistence")
	flag.DurationVar(&cfg.snapWait, "snapshot-interval", 0, "periodic snapshot interval; 0 disables")
	flag.StringVar(&cfg.wal, "wal", "", "write-ahead log base path; empty means <snapshot>.wal, \"off\" disables logging")
	flag.Parse()

	if err := run(cfg); err != nil {
		log.Fatalf("mindb: %v", err)
	}
}

func run(cfg config) error {
	log.Printf("mindb %s (%s, %s/%s)", version, runtime.Version(), runtime.GOOS, runtime.GOARCH)

	// The kernel decides whether the cascade runs at all, so which one is live
	// is the first thing to know when a deployment's search latency looks wrong.
	log.Printf("kernel: name=%s fast_int8=%t goarch=%s",
		math.KernelName(), math.HasFastInt8(), runtime.GOARCH)

	if cfg.snapWait > 0 && cfg.snapPath == "" {
		return errors.New("-snapshot-interval requires -snapshot")
	}
	walBase, err := walPath(cfg)
	if err != nil {
		return err
	}

	// Built before the engine because the fault callback closes over it. A
	// health server starts out SERVING, which is right: nothing accepts a
	// connection until the engine below has loaded.
	healthSrv := health.NewServer()
	healthSrv.SetServingStatus(serviceName, healthpb.HealthCheckResponse_SERVING)

	engine, recovery, err := core.Open(core.Options{
		Dims:     cfg.dims,
		Capacity: cfg.capacity,
		Snapshot: cfg.snapPath,
		WAL:      walBase,
		OnWALFault: func() {
			// Keep serving: what is in memory is still correct and still
			// useful. Stop advertising, though, because writes from here on
			// are acknowledged without being durable.
			log.Printf("write-ahead log failed; still serving, reporting NOT_SERVING so traffic drains")
			healthSrv.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
			healthSrv.SetServingStatus(serviceName, healthpb.HealthCheckResponse_NOT_SERVING)
		},
	})
	if err != nil {
		return err
	}

	logBoot(cfg, walBase, engine, recovery)

	lis, err := net.Listen("tcp", cfg.addr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", cfg.addr, err)
	}

	// Without ForceServerCodec the handlers, which return *flatbuffers.Builder,
	// fail at request time rather than at startup. Registering it here is not
	// optional.
	srv := grpc.NewServer(grpc.ForceServerCodec(flatbuffers.FlatbuffersCodec{}))
	mindb.RegisterVectorServiceServer(srv, api.New(engine, cfg.snapPath))

	// The health service speaks protobuf, and the codec above is forced for
	// every service on the server it is set on, so health cannot share this one:
	// the flatbuffers codec type-asserts what it is handed to *flatbuffers.Builder
	// and would panic on a health response. Hence a second listener, default codec.
	healthGRPC, err := serveHealth(cfg.healthAddr, healthSrv)
	if err != nil {
		return err
	}

	stopPeriodic := startPeriodicSnapshots(engine, cfg.snapPath, cfg.snapWait)

	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	errc := make(chan error, 1)
	go func() {
		log.Printf("listening on %s", cfg.addr)
		errc <- srv.Serve(lis)
	}()

	select {
	case err := <-errc:
		return err
	case sig := <-shutdown:
		log.Printf("received %s, shutting down", sig)
	}

	// NOT_SERVING before the drain, so the orchestrator takes this pod out of
	// rotation while in-flight requests finish rather than after.
	healthSrv.Shutdown()

	close(stopPeriodic)
	srv.GracefulStop()
	if healthGRPC != nil {
		healthGRPC.GracefulStop()
	}

	// Final snapshot after GracefulStop, so no in-flight write is missed. It
	// also retires the log, which is why Close comes after it and not before.
	if cfg.snapPath != "" {
		if err := engine.Save(cfg.snapPath); err != nil {
			return fmt.Errorf("final snapshot: %w", err)
		}
		log.Printf("final snapshot written to %s (%d vectors)", cfg.snapPath, engine.Len())
	}
	return engine.Close()
}

// walPath resolves -wal against -snapshot.
//
// Logging is on wherever persistence is, because a snapshot on its own silently
// loses every write taken since it was written. "off" is how to say that losing
// them is acceptable.
func walPath(cfg config) (string, error) {
	switch {
	case cfg.wal == "off":
		return "", nil
	case cfg.snapPath == "":
		if cfg.wal != "" {
			return "", errors.New("-wal requires -snapshot: without snapshots nothing retires log segments")
		}
		return "", nil
	case cfg.wal == "":
		return cfg.snapPath + ".wal", nil
	default:
		return cfg.wal, nil
	}
}

// serveHealth starts the gRPC health service on its own listener. An empty
// address returns a nil server and no health endpoint.
func serveHealth(addr string, srv *health.Server) (*grpc.Server, error) {
	if addr == "" {
		log.Printf("health service disabled")
		return nil, nil
	}
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("listen on %s for health: %w", addr, err)
	}
	g := grpc.NewServer()
	healthpb.RegisterHealthServer(g, srv)
	go func() {
		if err := g.Serve(lis); err != nil {
			log.Printf("health service stopped: %v", err)
		}
	}()
	log.Printf("health service on %s (grpc.health.v1.Health, service %q)", addr, serviceName)
	return g, nil
}

// logBoot reports what the engine came back with, including anything about the
// durable state an operator should see without having to ask for it.
func logBoot(cfg config, walBase string, engine *core.Engine, rec core.WALRecovery) {
	stats := engine.Stats()
	if cfg.snapPath == "" {
		log.Printf("persistence disabled: no -snapshot")
	} else if stats.Dims != cfg.dims {
		log.Printf("snapshot declares dims=%d, overriding -dims=%d", stats.Dims, cfg.dims)
	}
	log.Printf("engine ready: dims=%d capacity=%d loaded=%d approx_ram=%s",
		stats.Dims, stats.Capacity, stats.Count, humanBytes(stats.MemoryBytes))

	if walBase == "" {
		if cfg.snapPath != "" {
			log.Printf("WARNING: write-ahead log disabled; writes since the last snapshot are lost on an unclean stop")
		}
		return
	}

	log.Printf("write-ahead log: %s.?????? (%d segments, %d records replayed)",
		walBase, rec.Segments, rec.Records)
	if rec.Truncated {
		log.Printf("WARNING: the log ended in a damaged record; that record and everything after it was discarded")
	}
	if rec.LegacyMeta {
		log.Printf("WARNING: %s has no companion .meta file, so the log could not be checked against it; the next snapshot writes one",
			cfg.snapPath)
	}
	if cfg.snapWait == 0 {
		log.Printf("WARNING: -snapshot-interval is 0, so nothing retires log segments before shutdown and the log grows unbounded")
	}
}

func startPeriodicSnapshots(engine *core.Engine, path string, every time.Duration) chan struct{} {
	stop := make(chan struct{})
	if every <= 0 || path == "" {
		return stop
	}
	go func() {
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				if err := engine.Save(path); err != nil {
					log.Printf("periodic snapshot failed: %v", err)
				}
			}
		}
	}()
	return stop
}

// humanBytes formats a byte count for the startup log.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n/div >= unit && exp < 3 {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGT"[exp])
}
