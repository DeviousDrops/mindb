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

	"github.com/typicallhavok/mindb/pkg/api"
	"github.com/typicallhavok/mindb/pkg/core"
	"github.com/typicallhavok/mindb/pkg/math"
	"github.com/typicallhavok/mindb/pkg/mindb"
)

func main() {
	var (
		addr     = flag.String("addr", ":50051", "gRPC listen address")
		dims     = flag.Int("dims", 768, "vector dimension (ignored when a snapshot is loaded)")
		capacity = flag.Int("capacity", 100_000, "maximum number of vectors; memory is reserved eagerly at boot")
		snapPath = flag.String("snapshot", "", "snapshot file path; empty disables persistence")
		snapWait = flag.Duration("snapshot-interval", 0, "periodic snapshot interval; 0 disables")
	)
	flag.Parse()

	if err := run(*addr, *dims, *capacity, *snapPath, *snapWait); err != nil {
		log.Fatalf("mindb: %v", err)
	}
}

func run(addr string, dims, capacity int, snapPath string, snapWait time.Duration) error {
	// The kernel decides whether the cascade runs at all, so which one is live
	// is the first thing to know when a deployment's search latency looks wrong.
	log.Printf("kernel: name=%s fast_int8=%t goarch=%s",
		math.KernelName(), math.HasFastInt8(), runtime.GOARCH)

	engine, err := open(dims, capacity, snapPath)
	if err != nil {
		return err
	}
	log.Printf("engine ready: dims=%d capacity=%d loaded=%d approx_ram=%s",
		engine.Dims(), engine.Cap(), engine.Len(), humanBytes(estimateRAM(engine)))

	if snapWait > 0 && snapPath == "" {
		return errors.New("-snapshot-interval requires -snapshot")
	}

	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", addr, err)
	}

	// Without ForceServerCodec the handlers, which return *flatbuffers.Builder,
	// fail at request time rather than at startup. Registering it here is not
	// optional.
	srv := grpc.NewServer(grpc.ForceServerCodec(flatbuffers.FlatbuffersCodec{}))
	mindb.RegisterVectorServiceServer(srv, api.New(engine, snapPath))

	stopPeriodic := startPeriodicSnapshots(engine, snapPath, snapWait)

	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	errc := make(chan error, 1)
	go func() {
		log.Printf("listening on %s", addr)
		errc <- srv.Serve(lis)
	}()

	select {
	case err := <-errc:
		return err
	case sig := <-shutdown:
		log.Printf("received %s, shutting down", sig)
	}

	close(stopPeriodic)
	srv.GracefulStop()

	// Final snapshot after GracefulStop, so no in-flight write is missed.
	if snapPath != "" {
		if err := engine.Save(snapPath); err != nil {
			return fmt.Errorf("final snapshot: %w", err)
		}
		log.Printf("final snapshot written to %s (%d vectors)", snapPath, engine.Len())
	}
	return nil
}

// open loads the snapshot at path, or builds an empty engine if there is none.
func open(dims, capacity int, path string) (*core.Engine, error) {
	if path == "" {
		return core.New(dims, capacity)
	}

	engine, err := core.Load(path, capacity)
	switch {
	case err == nil:
		if engine.Dims() != dims {
			log.Printf("snapshot declares dims=%d, overriding -dims=%d", engine.Dims(), dims)
		}
		log.Printf("loaded %d vectors from %s", engine.Len(), path)
		return engine, nil
	case errors.Is(err, os.ErrNotExist):
		log.Printf("no snapshot at %s, starting empty", path)
		return core.New(dims, capacity)
	default:
		// Refuse to start rather than silently discarding a corrupt snapshot:
		// booting empty here would look like data loss and be indistinguishable
		// from a first run.
		return nil, fmt.Errorf("load snapshot %s: %w", path, err)
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

// estimateRAM reports the bytes reserved for vector storage. Payloads are not
// counted; they are caller-sized and allocated on demand.
func estimateRAM(e *core.Engine) int64 {
	// Per vector: dims*4 (float32) + dims*1 (int8 code) + 4 (scale) + 4 (residual).
	return int64(e.Cap()) * (int64(e.Dims())*5 + 8)
}

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
