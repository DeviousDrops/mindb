//go:build arm64

package math

// arm64 has no SIMD int8 kernel yet, so DotInt8 is the pure-Go loop and the
// cascade stays off. Correct, just not fast: ARM deployments run the plain
// float32 scan.
//
// The kernel that belongs here is a NEON SDOT/UDOT implementation — ARMv8.2-A
// dotprod, which the Ampere Altra parts MinDB is deployed on do have. That is a
// separate track; this file exists so the dispatch is already wired for it and
// so "arm64 falls back" is a decision rather than a fall-through. See
// DECISIONS.md, "Portability".
const (
	hasFastInt8 = false
	kernelName  = "pure-go"
)

var dotInt8Impl = dotInt8Generic
