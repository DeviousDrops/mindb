// Command mindb-load drives a running mindb-server with concurrent inserts and
// searches, so the metrics endpoint has something to show.
//
// The default run ramps the writer count 1, 2, 4, ... up to -writers, holding
// each step for -step. On the dashboard that draws a staircase: request rate
// and group-commit batch size climb with each step while fsync latency stays
// roughly flat, which is the whole argument for group commit in one picture.
//
//	mindb-load -addr localhost:50051 -dims 768 -writers 64 -step 20s
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"time"

	flatbuffers "github.com/google/flatbuffers/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/DeviousDrops/mindb/pkg/mindb"
)

type config struct {
	addr      string
	dims      int
	writers   int
	searchers int
	keyspace  int
	clusters  int
	topK      int
	step      time.Duration
	ramp      bool
	duration  time.Duration
}

func main() {
	var cfg config
	flag.StringVar(&cfg.addr, "addr", "localhost:50051", "mindb-server gRPC address")
	flag.IntVar(&cfg.dims, "dims", 768, "vector dimension; must match the server")
	flag.IntVar(&cfg.writers, "writers", 64, "concurrent writers (the ramp's ceiling)")
	flag.IntVar(&cfg.searchers, "searchers", 2, "concurrent searchers")
	flag.IntVar(&cfg.keyspace, "keyspace", 20000, "distinct ids; writes past it overwrite, so keep it under the server's -capacity")
	flag.IntVar(&cfg.clusters, "clusters", 64, "clusters the vectors are drawn around; clustered data is what makes the cascade's pruning realistic")
	flag.IntVar(&cfg.topK, "k", 10, "top-k per search")
	flag.BoolVar(&cfg.ramp, "ramp", true, "double the writer count every -step, from 1 up to -writers")
	flag.DurationVar(&cfg.step, "step", 20*time.Second, "how long each ramp step lasts")
	flag.DurationVar(&cfg.duration, "duration", 0, "total run time with -ramp=false; 0 runs until interrupted")
	flag.Parse()

	if err := run(cfg); err != nil {
		log.Fatalf("mindb-load: %v", err)
	}
}

func run(cfg config) error {
	if cfg.writers < 1 || cfg.dims < 1 || cfg.keyspace < 1 || cfg.clusters < 1 {
		return fmt.Errorf("-writers, -dims, -keyspace and -clusters must be positive")
	}
	conn, err := grpc.NewClient(cfg.addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(flatbuffers.FlatbuffersCodec{})),
	)
	if err != nil {
		return err
	}
	defer conn.Close()
	client := mindb.NewVectorServiceClient(conn)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	centroids := make([][]float32, cfg.clusters)
	r := rand.New(rand.NewSource(1))
	for i := range centroids {
		centroids[i] = unit(randVec(r, cfg.dims))
	}

	var (
		active             atomic.Int64 // writers allowed to run right now
		inserts, searches  atomic.Int64
		insertErr, searchE atomic.Int64
		nextID             atomic.Int64
		wg                 sync.WaitGroup
	)
	if cfg.ramp {
		active.Store(1)
	} else {
		active.Store(int64(cfg.writers))
	}

	for w := 0; w < cfg.writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(w) + 100))
			for ctx.Err() == nil {
				if int64(w) >= active.Load() {
					time.Sleep(50 * time.Millisecond)
					continue
				}
				id := fmt.Sprintf("v%d", nextID.Add(1)%int64(cfg.keyspace))
				vec := near(r, centroids[r.Intn(len(centroids))], 0.08)
				cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
				_, err := client.Insert(cctx, buildInsert(id, vec))
				cancel()
				if err != nil {
					if ctx.Err() == nil {
						insertErr.Add(1)
					}
					continue
				}
				inserts.Add(1)
			}
		}(w)
	}
	for s := 0; s < cfg.searchers; s++ {
		wg.Add(1)
		go func(s int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(s) + 10_000))
			for ctx.Err() == nil {
				q := near(r, centroids[r.Intn(len(centroids))], 0.08)
				cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
				_, err := client.Search(cctx, buildSearch(q, int32(cfg.topK)))
				cancel()
				if err != nil {
					if ctx.Err() == nil {
						searchE.Add(1)
					}
					continue
				}
				searches.Add(1)
			}
		}(s)
	}

	// Ramp controller.
	go func() {
		if !cfg.ramp {
			if cfg.duration > 0 {
				select {
				case <-ctx.Done():
				case <-time.After(cfg.duration):
					stop()
				}
			}
			return
		}
		for n := int64(1); ; n *= 2 {
			if n > int64(cfg.writers) {
				n = int64(cfg.writers)
			}
			active.Store(n)
			log.Printf("ramp: %d writers for %s", n, cfg.step)
			select {
			case <-ctx.Done():
				return
			case <-time.After(cfg.step):
			}
			if n == int64(cfg.writers) {
				log.Printf("ramp: done")
				stop()
				return
			}
		}
	}()

	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	var lastI, lastS int64
	for {
		select {
		case <-ctx.Done():
			wg.Wait()
			log.Printf("total: %d inserts, %d searches, %d insert errors, %d search errors",
				inserts.Load(), searches.Load(), insertErr.Load(), searchE.Load())
			return nil
		case <-tick.C:
			i, s := inserts.Load(), searches.Load()
			log.Printf("writers=%-3d inserts/s=%-6d searches/s=%-5d errors=%d",
				active.Load(), i-lastI, s-lastS, insertErr.Load()+searchE.Load())
			lastI, lastS = i, s
		}
	}
}

func randVec(r *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

func unit(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	n = math.Sqrt(n)
	for i := range v {
		v[i] = float32(float64(v[i]) / n)
	}
	return v
}

// near returns a point within spread of c in every coordinate.
func near(r *rand.Rand, c []float32, spread float32) []float32 {
	v := make([]float32, len(c))
	for i := range v {
		v[i] = c[i] + (r.Float32()*2-1)*spread
	}
	return v
}

func buildInsert(id string, vals []float32) *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(len(vals)*4 + 64)
	idOff := b.CreateString(id)
	mindb.VectorStartValuesVector(b, len(vals))
	for i := len(vals) - 1; i >= 0; i-- {
		b.PrependFloat32(vals[i])
	}
	valsOff := b.EndVector(len(vals))

	mindb.VectorStart(b)
	mindb.VectorAddId(b, idOff)
	mindb.VectorAddValues(b, valsOff)
	vec := mindb.VectorEnd(b)

	mindb.InsertRequestStartVectorsVector(b, 1)
	b.PrependUOffsetT(vec)
	vecs := b.EndVector(1)

	mindb.InsertRequestStart(b)
	mindb.InsertRequestAddVectors(b, vecs)
	b.Finish(mindb.InsertRequestEnd(b))
	return b
}

func buildSearch(query []float32, topK int32) *flatbuffers.Builder {
	b := flatbuffers.NewBuilder(len(query)*4 + 32)
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
