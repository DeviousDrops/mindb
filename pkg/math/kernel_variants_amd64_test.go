//go:build amd64

package math

// archKernelVariants exposes the Avo-generated AVX2 kernel to the cross-check,
// bypassing dispatch so the kernel is tested even when it is not the one
// selected. When the CPU lacks AVX2 the kernel must not be called at all, so
// there is nothing to add.
func archKernelVariants() []kernelVariant {
	if !hasFastInt8 {
		return nil
	}
	return []kernelVariant{{"avx2", dotInt8AVX2}}
}
