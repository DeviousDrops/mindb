package math

import (
	stdmath "math"
	"math/rand"
	"testing"
)

// kernelVariant is one DotInt8 implementation available on this build.
type kernelVariant struct {
	name string
	fn   func(q []float32, code []int8) float32
}

// kernelVariants lists every implementation the current GOARCH can reach: the
// one DotInt8 actually dispatches to, plus whatever SIMD kernels the
// architecture compiled in (see kernel_variants_*_test.go).
//
// On architectures with no SIMD kernel this is just the pure-Go loop compared
// against itself. That is deliberately tautological today; its job is to fail
// the moment a NEON kernel is added or the dispatch wiring breaks, without
// anyone having to remember to write the test then.
func kernelVariants() []kernelVariant {
	all := []kernelVariant{{"dispatch:" + kernelName, dotInt8Impl}}
	return append(all, archKernelVariants()...)
}

// crosscheckDims straddle every block and tail boundary the kernels have: the
// 8-wide pure-Go unroll, the AVX2 kernel's 32-element block, and the 768 real
// queries use.
var crosscheckDims = []int{0, 1, 7, 8, 9, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 768, 769}

// TestKernelsMatchGenericReference is the portability contract: whatever
// DotInt8 dispatches to on this machine must agree with the pure-Go reference.
// dotInt8Generic is the oracle — it is the definition of a correct result, and
// every other path is an optimization of it.
func TestKernelsMatchGenericReference(t *testing.T) {
	r := rand.New(rand.NewSource(7))

	for _, v := range kernelVariants() {
		for _, dims := range crosscheckDims {
			q := randVec(r, dims)
			code := randCodes(r, dims)

			want := dotInt8Generic(q, code)
			got := v.fn(q, code)

			if d := stdmath.Abs(float64(got - want)); d > dotTolerance(q, code) {
				t.Errorf("%s: dims=%d: got %v, want %v (diff %g, tol %g)",
					v.name, dims, got, want, d, dotTolerance(q, code))
			}
		}
	}
}

// dotTolerance bounds how far two float32 dot products over the same inputs may
// legitimately differ.
//
// The bound scales with Σ|qᵢ·codeᵢ|, not with the result. A dot product of
// mixed-sign terms cancels, so the result can be near zero while the terms
// summed to get there are large; a tolerance relative to the result would then
// demand precision no float32 accumulation can deliver. Σ|qᵢ·codeᵢ| is the
// quantity the rounding error actually scales with.
//
// 1e-5 is roughly twice the worst case for the 8-accumulator pure-Go loop
// (⌈n/8⌉ additions at a 2⁻²⁴ unit roundoff); wider kernels accumulate less
// error, not more. The additive floor keeps dims=0, where the sum is empty,
// from demanding bit-exactness against a tolerance of zero.
func dotTolerance(q []float32, code []int8) float64 {
	var sumAbs float64
	for i := range q {
		sumAbs += stdmath.Abs(float64(q[i]) * float64(code[i]))
	}
	return 1e-5*sumAbs + 1e-6
}

func randCodes(r *rand.Rand, n int) []int8 {
	code := make([]int8, n)
	for i := range code {
		code[i] = int8(r.Intn(255) - 127)
	}
	return code
}
