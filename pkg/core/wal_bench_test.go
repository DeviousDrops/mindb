package core

import (
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// BenchmarkWALInsert measures what durability costs and what group commit buys
// back.
//
// Three arms. "off" is the engine with no log at all, which is the ceiling.
// "fsync-per-write" makes every writer pay for its own fsync, which is what the
// log would do without batching. "group-commit" is the shipped path, where one
// writer flushes and syncs for everyone who arrived while the last sync was in
// flight.
//
// At one writer the last two are the same benchmark: there is nobody to batch
// with, and the number is simply the cost of an fsync on whatever disk the
// benchmark is running on. The gap opens with concurrency, and how far it opens
// is a property of the disk, not of this code -- expect a different answer on an
// NVMe SSD than on a cloud volume with a network in the way.
func BenchmarkWALInsert(b *testing.B) {
	const (
		dims = 64
		// Ids cycle through a fixed set, so the slab stays small no matter how
		// many iterations the benchmark runs. Overwriting an id exercises the
		// same write path as inserting a new one.
		ring = 4096
	)

	ids := make([]string, ring)
	for i := range ids {
		ids[i] = fmt.Sprintf("k%04d", i)
	}
	vec := make([]float32, dims)
	for i := range vec {
		vec[i] = float32(i%13) + 1
	}
	payload := make([]byte, 64)

	for _, arm := range []string{"off", "fsync-per-write", "group-commit"} {
		for _, writers := range []int{1, 8, 64} {
			b.Run(fmt.Sprintf("%s/writers=%d", arm, writers), func(b *testing.B) {
				dir := b.TempDir()
				opts := Options{Dims: dims, Capacity: ring}
				if arm != "off" {
					opts.Snapshot = filepath.Join(dir, "snap.mindb")
					opts.WAL = filepath.Join(dir, "snap.mindb.wal")
				}

				e, _, err := Open(opts)
				if err != nil {
					b.Fatal(err)
				}
				defer e.Close()
				if arm == "fsync-per-write" {
					e.wal.syncAlone = true
				}

				var next atomic.Int64
				var wg sync.WaitGroup

				b.ResetTimer()
				for w := 0; w < writers; w++ {
					wg.Add(1)
					go func() {
						defer wg.Done()
						for {
							n := next.Add(1) - 1
							if n >= int64(b.N) {
								return
							}
							if err := e.Insert(ids[n%ring], vec, payload); err != nil {
								b.Error(err)
								return
							}
						}
					}()
				}
				wg.Wait()
				b.StopTimer()
			})
		}
	}
}
