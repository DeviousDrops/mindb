package core

import (
	"runtime"
	"sync"

	"github.com/DeviousDrops/mindb/pkg/math"
)

// guardFraction is the survivor rate above which the cascade abandons its
// shortlist and runs a plain float32 scan instead.
//
// Measured by BenchmarkGuardCrossover at 20k x 768 on this machine: scattered
// rescoring alone equals a full parallel scan at f ≈ 49%. That is the ceiling,
// and it is well under the ~75% a bytes-moved argument predicts, for two
// reasons — rescoring walks the float32 array out of order, and pass 3 is
// serial while the scan is parallel.
//
// The real threshold is that ceiling minus the bound pass, which cannot be
// measured honestly until the AVX2 kernel exists: at a quarter of a float32
// scan the crossover is ~34%, at half it is ~22%. 0.25 sits inside that range,
// deliberately at the cautious end, because setting it too low costs only the
// thin margin near the crossover while setting it too high means paying for
// both passes. Re-measure in stage 3.
//
// Getting it wrong is bounded either way: results are identical, only the time
// changes. In practice it is nearly unreachable — clustered data measures 0.4%
// survivors, two orders of magnitude below the threshold.
var guardFraction = 0.25

// searchCascade returns the top k using bound-and-refine, and reports how many
// vectors survived pruning. A survivor count equal to the live count means the
// guard fired and a full float32 scan ran instead.
//
// The algorithm, and why it cannot drop a correct answer:
//
//  1. Score every vector against its int8 code. With a float32 query the only
//     error term is that vector's residual, so [approx−ρ, approx+ρ] is a hard
//     bound on the true score.
//  2. Let τ be the k-th largest lower bound. At least k vectors have a true
//     score ≥ τ, so the k-th largest true score is itself ≥ τ.
//  3. Therefore every member of the true top-k has a true score ≥ τ, and since
//     hi ≥ true, every member has hi ≥ τ. Discarding hi < τ is safe.
//  4. Rescore the survivors exactly.
//
// Caller must hold at least the read lock.
func (e *Engine) searchCascade(q []float32, k int) (*topK, int) {
	hw := int(e.highWater)

	approxPtr := e.scratch.Get().(*[]float32)
	defer e.scratch.Put(approxPtr)
	approx := (*approxPtr)[:hw]

	// Pass 1: the expensive one. Score every live slot in int8 and collect the
	// k largest lower bounds.
	bounds := e.scanBounds(q, k, approx, hw)

	tau := float32(-2) // below any achievable cosine similarity
	if len(bounds.heap) == bounds.k {
		tau = bounds.heap[0].score
	}
	e.heaps.Put(bounds)

	// Pass 2: cheap. Two float32 reads per slot, no vector data touched.
	survivors := make([]uint32, 0, 256)
	for slot := 0; slot < hw; slot++ {
		if !e.live[slot] {
			continue
		}
		if approx[slot]+e.residuals[slot] >= tau {
			survivors = append(survivors, uint32(slot))
		}
	}

	if float64(len(survivors)) > guardFraction*float64(e.count) {
		return e.scan(q, k), e.count
	}

	best := e.takeHeap(k)
	dims := e.dims
	for _, slot := range survivors {
		base := int(slot) * dims
		best.push(candidate{
			slot:  slot,
			score: math.Dot(q, e.vectors[base:base+dims]),
		})
	}
	return best, len(survivors)
}

// scanBounds fills approx with per-slot approximate scores and returns a heap of
// the k largest lower bounds.
func (e *Engine) scanBounds(q []float32, k int, approx []float32, hw int) *topK {
	workers := runtime.GOMAXPROCS(0)
	if hw < parallelScanThreshold || workers < 2 {
		h := e.takeHeap(k)
		e.boundRange(q, 0, hw, approx, h)
		return h
	}

	chunk := (hw + workers - 1) / workers
	heaps := make([]*topK, 0, workers)

	var wg sync.WaitGroup
	for lo := 0; lo < hw; lo += chunk {
		hi := lo + chunk
		if hi > hw {
			hi = hw
		}
		h := e.takeHeap(k)
		heaps = append(heaps, h)
		wg.Add(1)
		go func(lo, hi int, h *topK) {
			defer wg.Done()
			e.boundRange(q, lo, hi, approx, h)
		}(lo, hi, h)
	}
	wg.Wait()

	merged := heaps[0]
	for _, h := range heaps[1:] {
		for _, c := range h.heap {
			merged.push(c)
		}
		e.heaps.Put(h)
	}
	return merged
}

// boundRange scores slots in [lo, hi) against their int8 codes. Workers write
// disjoint ranges of approx, so no synchronization is needed.
func (e *Engine) boundRange(q []float32, lo, hi int, approx []float32, h *topK) {
	dims := e.dims
	for slot := lo; slot < hi; slot++ {
		if !e.live[slot] {
			continue
		}
		base := slot * dims
		a := math.DotInt8(q, e.codes[base:base+dims]) * e.scales[slot]
		approx[slot] = a
		h.push(candidate{slot: uint32(slot), score: a - e.residuals[slot]})
	}
}
