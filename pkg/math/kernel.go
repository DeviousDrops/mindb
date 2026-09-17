package math

// HasFastInt8 reports whether DotInt8 is backed by a SIMD kernel.
//
// This gates the cascade. Pure-Go int8 plateaus around 0.83 G MAC/s — 7.2x short
// of the memory wall and slower than simply scanning float32 — because Go cannot
// emit the multiply-accumulate instructions that make int8 pay off. Taking the
// cascade without a fast kernel would move a quarter of the bytes and still lose.
//
// The value is set per architecture at build time: amd64 detects AVX2+FMA at
// init and falls back if they are absent, arm64 and everything else are pure
// Go. See kernel_amd64.go, kernel_arm64.go, kernel_generic.go.
func HasFastInt8() bool { return hasFastInt8 }

// KernelName reports which DotInt8 implementation is active, for startup
// logging and benchmark labelling.
func KernelName() string { return kernelName }
