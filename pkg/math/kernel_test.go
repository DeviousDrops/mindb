package math

import "testing"

// TestKernelNameReflectsDispatch keeps the reported name honest on every
// architecture: it is what startup logging and benchmark labels claim is
// running, so a stale name misattributes results to the wrong kernel.
func TestKernelNameReflectsDispatch(t *testing.T) {
	if kernelName == "" {
		t.Fatal("kernelName is empty")
	}
	if hasFastInt8 && kernelName == "pure-go" {
		t.Errorf("hasFastInt8=true but kernelName=%q", kernelName)
	}
	if !hasFastInt8 && kernelName != "pure-go" {
		t.Errorf("hasFastInt8=false but kernelName=%q, want pure-go", kernelName)
	}
	if dotInt8Impl == nil {
		t.Error("dotInt8Impl is nil; DotInt8 would panic")
	}
}
