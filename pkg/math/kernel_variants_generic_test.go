//go:build !amd64 && !arm64

package math

// archKernelVariants is empty: these architectures have the pure-Go loop only.
func archKernelVariants() []kernelVariant { return nil }
