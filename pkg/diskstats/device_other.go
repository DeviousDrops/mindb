//go:build !linux

package diskstats

// DeviceFor needs Linux: the device number comes from stat(2) and the counters
// from /proc. Elsewhere the server logs this and runs without disk metrics.
func DeviceFor(path string) ([]Device, error) {
	return nil, ErrUnsupported
}
