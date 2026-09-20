package core

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/DeviousDrops/mindb/pkg/math"
)

// clusteredVec builds vectors that sit in tight clusters.
//
// Uniform-random vectors in high dimensions are nearly orthogonal, which makes
// any pruning scheme look brilliant: the top-k stands far above the rest and the
// bound separates them trivially. Clustered data is the honest case, because the
// competitors for the top-k are all crowded together — which is exactly where a
// bound has to be tight to be useful.
func clusteredVec(r *rand.Rand, centroids [][]float32, spread float32) []float32 {
	c := centroids[r.Intn(len(centroids))]
	v := make([]float32, len(c))
	for i := range v {
		v[i] = c[i] + (r.Float32()*2-1)*spread
	}
	return v
}

func makeCentroids(r *rand.Rand, n, dims int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		out[i] = randVec(r, dims)
		math.Normalize(out[i])
	}
	return out
}

// buildClustered returns an engine full of clustered vectors plus a query source.
func buildClustered(t testing.TB, dims, n, clusters int, seed int64) (*Engine, [][]float32, *rand.Rand) {
	t.Helper()
	r := rand.New(rand.NewSource(seed))
	centroids := makeCentroids(r, clusters, dims)

	e, err := New(dims, n)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := e.Insert(fmt.Sprintf("v%06d", i), clusteredVec(r, centroids, 0.1), nil); err != nil {
			t.Fatal(err)
		}
	}
	return e, centroids, r
}

// assertIdentical is the exactness guarantee, spelled out. Not "close", not
// "same recall" — the same ids and the same bits.
func assertIdentical(t *testing.T, label string, exact, cascade []Result) {
	t.Helper()
	if len(exact) != len(cascade) {
		t.Fatalf("%s: cascade returned %d results, brute force returned %d",
			label, len(cascade), len(exact))
	}
	for i := range exact {
		if exact[i].ID != cascade[i].ID {
			t.Fatalf("%s rank %d: cascade %q (%v), brute force %q (%v)",
				label, i, cascade[i].ID, cascade[i].Score, exact[i].ID, exact[i].Score)
		}
		if exact[i].Score != cascade[i].Score {
			t.Fatalf("%s rank %d (%s): cascade score %v, brute force %v — not bit-identical",
				label, i, exact[i].ID, cascade[i].Score, exact[i].Score)
		}
	}
}

// TestCascadeIsExact is the centerpiece of the whole project. If it fails, the
// central claim — ANN-class pruning with exact-kNN guarantees — is false.
func TestCascadeIsExact(t *testing.T) {
	const (
		dims     = 128
		n        = 4000
		clusters = 12
		queries  = 500
	)
	e, centroids, r := buildClustered(t, dims, n, clusters, 1)

	for _, k := range []int{1, 5, 10, 50} {
		for i := 0; i < queries; i++ {
			// Half the queries sit inside a cluster, where the top-k are
			// tightly packed and the bound has the least room to work.
			var q []float32
			if i%2 == 0 {
				q = clusteredVec(r, centroids, 0.1)
			} else {
				q = randVec(r, dims)
			}

			e.SetCascade(false)
			exact, err := e.Search(q, k)
			if err != nil {
				t.Fatal(err)
			}
			e.SetCascade(true)
			cascade, err := e.Search(q, k)
			if err != nil {
				t.Fatal(err)
			}
			assertIdentical(t, fmt.Sprintf("k=%d query=%d", k, i), exact, cascade)
		}
	}
}

func TestCascadeIsExactWithDeletes(t *testing.T) {
	const dims, n = 64, 3000
	e, centroids, r := buildClustered(t, dims, n, 8, 2)

	// Holes in the slot array plus a populated free list: the cascade's two
	// passes must agree with each other about which slots are live.
	for i := 0; i < n; i += 3 {
		_, _ = e.Delete(fmt.Sprintf("v%06d", i))
	}
	for i := 0; i < 200; i++ {
		if err := e.Insert(fmt.Sprintf("new%04d", i), clusteredVec(r, centroids, 0.1), nil); err != nil {
			t.Fatal(err)
		}
	}

	for i := 0; i < 200; i++ {
		q := clusteredVec(r, centroids, 0.1)

		e.SetCascade(false)
		exact, err := e.Search(q, 10)
		if err != nil {
			t.Fatal(err)
		}
		e.SetCascade(true)
		cascade, err := e.Search(q, 10)
		if err != nil {
			t.Fatal(err)
		}
		assertIdentical(t, fmt.Sprintf("query=%d", i), exact, cascade)

		for _, hit := range cascade {
			var idx int
			if _, err := fmt.Sscanf(hit.ID, "v%06d", &idx); err == nil && idx%3 == 0 {
				t.Fatalf("deleted vector %q returned by cascade", hit.ID)
			}
		}
	}
}

// TestCascadeIsExactAtScale crosses the parallel threshold, so the bound pass
// runs across workers writing disjoint ranges of the scratch array.
func TestCascadeIsExactAtScale(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large exactness test in -short mode")
	}
	const dims = 96
	n := parallelScanThreshold + 4000
	e, centroids, r := buildClustered(t, dims, n, 20, 3)

	if int(e.highWater) < parallelScanThreshold {
		t.Fatalf("did not reach the parallel path: highWater=%d", e.highWater)
	}

	for i := 0; i < 150; i++ {
		q := clusteredVec(r, centroids, 0.1)

		e.SetCascade(false)
		exact, err := e.Search(q, 20)
		if err != nil {
			t.Fatal(err)
		}
		e.SetCascade(true)
		cascade, err := e.Search(q, 20)
		if err != nil {
			t.Fatal(err)
		}
		assertIdentical(t, fmt.Sprintf("query=%d", i), exact, cascade)
	}
}

// TestCascadeSurvivesGuardFallback pins down that the adaptive guard changes
// only the time taken, never the answer.
func TestCascadeSurvivesGuardFallback(t *testing.T) {
	const dims, n = 64, 2000
	e, centroids, r := buildClustered(t, dims, n, 6, 4)

	original := guardFraction
	defer func() { guardFraction = original }()

	// A guard of 0 forces the fallback on every query; 1 disables it entirely.
	for _, g := range []float64{0, 0.2, 1} {
		guardFraction = g
		for i := 0; i < 100; i++ {
			q := clusteredVec(r, centroids, 0.1)

			e.SetCascade(false)
			exact, err := e.Search(q, 10)
			if err != nil {
				t.Fatal(err)
			}
			e.SetCascade(true)
			cascade, err := e.Search(q, 10)
			if err != nil {
				t.Fatal(err)
			}
			assertIdentical(t, fmt.Sprintf("guard=%v query=%d", g, i), exact, cascade)
		}
	}
}

// TestCascadeBoundContainsTrueScore checks the invariant directly rather than
// through its consequences. If this fails the pruning is unsound even when the
// end-to-end results happen to agree.
func TestCascadeBoundContainsTrueScore(t *testing.T) {
	const dims, n = 128, 2000
	e, centroids, r := buildClustered(t, dims, n, 10, 5)

	for trial := 0; trial < 100; trial++ {
		q := clusteredVec(r, centroids, 0.1)
		math.Normalize(q)

		e.mu.RLock()
		for slot := 0; slot < int(e.highWater); slot++ {
			if !e.live[slot] {
				continue
			}
			base := slot * e.dims
			approx := math.DotInt8(q, e.codes[base:base+e.dims]) * e.scales[slot]
			rho := e.residuals[slot]
			truth := math.Dot(q, e.vectors[base:base+e.dims])

			const eps = 1e-5
			if truth < approx-rho-eps || truth > approx+rho+eps {
				e.mu.RUnlock()
				t.Fatalf("trial %d slot %d: true score %v outside [%v, %v]",
					trial, slot, truth, approx-rho, approx+rho)
			}
		}
		e.mu.RUnlock()
	}
}

// TestCascadePruningIsEffective reports the survivor rate. Exactness is proved
// elsewhere; this is about whether the pruning is worth doing at all, since a
// valid bound that prunes nothing is exactly what killed the 4-bit variant.
func TestCascadePruningIsEffective(t *testing.T) {
	const dims, n, queries = 128, 5000, 200
	e, centroids, r := buildClustered(t, dims, n, 15, 6)

	original := guardFraction
	guardFraction = 1 // measure raw pruning, not the guard
	defer func() { guardFraction = original }()

	total := 0
	for i := 0; i < queries; i++ {
		q := clusteredVec(r, centroids, 0.1)
		math.Normalize(q)

		e.mu.RLock()
		h, survivors := e.searchCascade(q, 10)
		e.mu.RUnlock()
		e.heaps.Put(h)
		total += survivors
	}

	rate := float64(total) / float64(queries) / float64(n) * 100
	t.Logf("mean survivors: %.1f of %d (%.3f%%) over %d clustered queries",
		float64(total)/float64(queries), n, rate, queries)

	if rate > 25 {
		t.Errorf("survivor rate %.2f%% is too high for the cascade to pay off", rate)
	}
}

func TestCascadeHandlesSmallAndEmptyCases(t *testing.T) {
	e, err := New(16, 100)
	if err != nil {
		t.Fatal(err)
	}
	e.SetCascade(true)

	q := make([]float32, 16)
	q[0] = 1

	if res, err := e.Search(q, 5); err != nil || res != nil {
		t.Fatalf("empty engine: got %v, %v", res, err)
	}

	// k larger than the population means the bound heap never fills, so tau
	// stays at its sentinel and nothing may be pruned.
	for i := 0; i < 3; i++ {
		v := make([]float32, 16)
		v[i] = 1
		if err := e.Insert(fmt.Sprintf("v%d", i), v, nil); err != nil {
			t.Fatal(err)
		}
	}
	res, err := e.Search(q, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 3 {
		t.Fatalf("got %d results, want 3", len(res))
	}
	if res[0].ID != "v0" {
		t.Errorf("rank 0 is %q, want v0", res[0].ID)
	}
}

// BenchmarkGuardCrossover measures the half of the guard equation that does not
// depend on the int8 kernel: scattered rescoring of a survivor set versus one
// sequential float32 sweep.
//
// The full cost model is cascade(f) = bound_pass + rescore(f) against scan for
// brute force. bound_pass cannot be measured honestly until the AVX2 kernel
// lands — in pure Go it is slower than the thing it is meant to avoid. rescore(f)
// and scan can be measured now, and they are what actually set the threshold,
// because scattered access over the float32 array is the term that misbehaves.
//
// Solve for f: the guard should fire when rescore(f) > scan − bound_pass.
func BenchmarkGuardCrossover(b *testing.B) {
	const dims, n = 768, 20000
	e, centroids, r := buildClustered(b, dims, n, 20, 8)
	q := clusteredVec(r, centroids, 0.1)
	math.Normalize(q)

	b.Run("fullscan", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h := e.scan(q, 10)
			e.heaps.Put(h)
		}
	})

	for _, pct := range []int{5, 10, 20, 30, 50, 75, 100} {
		// Survivors are slot numbers in ascending order but sparse, which is the
		// access pattern the real pass 3 produces.
		survivors := make([]uint32, 0, n)
		step := 100.0 / float64(pct)
		for x := 0.0; int(x) < n; x += step {
			survivors = append(survivors, uint32(x))
		}

		b.Run(fmt.Sprintf("rescore%d%%", pct), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				h := e.takeHeap(10)
				for _, slot := range survivors {
					base := int(slot) * dims
					h.push(candidate{slot: slot, score: math.Dot(q, e.vectors[base:base+dims])})
				}
				e.heaps.Put(h)
			}
		})
	}
}

func BenchmarkSearchPaths(b *testing.B) {
	const dims, n, k = 768, 20000, 10
	e, centroids, r := buildClustered(b, dims, n, 20, 7)
	q := clusteredVec(r, centroids, 0.1)

	for _, mode := range []struct {
		name    string
		cascade bool
	}{{"bruteforce", false}, {"cascade", true}} {
		e.SetCascade(mode.cascade)
		b.Run(mode.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := e.Search(q, k); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
