//go:build arm64

package math

// archKernelVariants is empty until a NEON SDOT kernel exists. When one lands,
// adding it here is all the cross-check needs.
func archKernelVariants() []kernelVariant { return nil }
