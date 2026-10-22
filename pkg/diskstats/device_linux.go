//go:build linux

package diskstats

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// DeviceFor returns the block device that holds path, followed by the whole
// disk it is a partition of, if it is one.
//
// Both are returned because they answer different questions. The partition is
// what MinDB's filesystem sits on. The whole disk is where the device's write
// cache is, so flush requests — the device-side half of an fsync — are counted
// there and may read zero on the partition.
//
// path must exist; pass the data directory, not a file that may not have been
// created yet.
func DeviceFor(path string) ([]Device, error) {
	var st unix.Stat_t
	if err := unix.Stat(path, &st); err != nil {
		return nil, fmt.Errorf("diskstats: stat %s: %w", path, err)
	}
	dev := uint64(st.Dev) // uint32 on some architectures, uint64 on most
	major, minor := unix.Major(dev), unix.Minor(dev)

	// Major 0 is the kernel's "unnamed" block major, handed to every
	// filesystem without a real device: overlayfs, tmpfs, proc, btrfs.
	if major == 0 {
		return nil, fmt.Errorf("%w: %s is on device 0:%d", ErrNoBlockDevice, path, minor)
	}

	f, err := os.Open("/proc/diskstats")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	_, name, err := Parse(f, major, minor)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNoBlockDevice, err)
	}
	devs := []Device{{Major: major, Minor: minor, Name: name}}

	if parent, ok := parentDisk(major, minor); ok {
		devs = append(devs, parent)
	}
	return devs, nil
}

// parentDisk resolves a partition to its whole disk through sysfs, where
// /sys/dev/block/M:m links to .../block/<disk>/<partition>.
func parentDisk(major, minor uint32) (Device, bool) {
	sys := fmt.Sprintf("/sys/dev/block/%d:%d", major, minor)
	if _, err := os.Stat(filepath.Join(sys, "partition")); err != nil {
		return Device{}, false // not a partition
	}
	real, err := filepath.EvalSymlinks(sys)
	if err != nil {
		return Device{}, false
	}
	parentDir := filepath.Dir(real)
	raw, err := os.ReadFile(filepath.Join(parentDir, "dev"))
	if err != nil {
		return Device{}, false
	}
	var pmaj, pmin uint32
	if _, err := fmt.Sscanf(string(raw), "%d:%d", &pmaj, &pmin); err != nil {
		return Device{}, false
	}
	return Device{Major: pmaj, Minor: pmin, Name: filepath.Base(parentDir)}, true
}
