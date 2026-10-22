// Package diskstats reads the kernel's block-device and per-process I/O
// counters: /proc/diskstats for the device MinDB's data lives on, and
// /proc/self/io for MinDB itself.
//
// The parsers work on any io.Reader and have no build constraints, so they are
// tested everywhere. Only finding which device a path is on needs Linux; on
// other systems DeviceFor returns ErrUnsupported and the server runs without
// disk metrics.
package diskstats

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// ErrUnsupported is returned where there is no /proc to read.
var ErrUnsupported = errors.New("diskstats: not supported on this platform")

// ErrNoBlockDevice means the path is on a filesystem with no block device
// under it — overlayfs, tmpfs, a network mount, or btrfs, which reports an
// anonymous device number. There is nothing in /proc/diskstats to read.
var ErrNoBlockDevice = errors.New("diskstats: path is not on a block device")

// SectorSize is the unit /proc/diskstats counts in. It is always 512,
// whatever the device's real sector size is.
const SectorSize = 512

// Device identifies a block device by the number /proc/diskstats keys it on.
type Device struct {
	Major, Minor uint32
	Name         string // e.g. "nvme0n1p2", as the kernel names it
}

func (d Device) String() string { return fmt.Sprintf("%s (%d:%d)", d.Name, d.Major, d.Minor) }

// Stats is one line of /proc/diskstats. Times are in milliseconds, as the
// kernel reports them. Documentation/admin-guide/iostats.rst is the reference.
type Stats struct {
	ReadsCompleted  uint64
	ReadsMerged     uint64
	SectorsRead     uint64
	ReadTimeMs      uint64
	WritesCompleted uint64
	WritesMerged    uint64
	SectorsWritten  uint64
	WriteTimeMs     uint64

	// InFlight is a gauge, not a counter: requests issued to the driver and
	// not yet completed.
	InFlight uint64

	// IOTimeMs is wall time the device had at least one request in flight;
	// its rate is utilization. WeightedIOTimeMs is that time multiplied by
	// the number in flight; its rate is the average queue depth.
	IOTimeMs         uint64
	WeightedIOTimeMs uint64

	// Discard fields, kernel 4.18+.
	HasDiscard        bool
	DiscardsCompleted uint64
	SectorsDiscarded  uint64
	DiscardTimeMs     uint64

	// Flush fields, kernel 5.5+. An fsync on a device with a volatile write
	// cache ends in a flush request, so this is where MinDB's fsyncs show up
	// at the device.
	HasFlush         bool
	FlushesCompleted uint64
	FlushTimeMs      uint64
}

// Parse finds major:minor in r, which has the format of /proc/diskstats.
func Parse(r io.Reader, major, minor uint32) (Stats, string, error) {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 3 {
			continue
		}
		maj, err1 := strconv.ParseUint(f[0], 10, 32)
		min, err2 := strconv.ParseUint(f[1], 10, 32)
		if err1 != nil || err2 != nil || uint32(maj) != major || uint32(min) != minor {
			continue
		}
		s, err := parseFields(f[3:])
		return s, f[2], err
	}
	if err := sc.Err(); err != nil {
		return Stats{}, "", err
	}
	return Stats{}, "", fmt.Errorf("diskstats: device %d:%d not listed", major, minor)
}

func parseFields(f []string) (Stats, error) {
	// 11 fields since 2.6, 15 with discards, 17 with flushes. Old kernels
	// listed partitions with only 4 fields; nothing that runs Go 1.25 does.
	if len(f) < 11 {
		return Stats{}, fmt.Errorf("diskstats: %d fields, want at least 11", len(f))
	}
	v := make([]uint64, len(f))
	for i, s := range f {
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return Stats{}, fmt.Errorf("diskstats: field %d: %w", i+4, err)
		}
		v[i] = n
	}
	s := Stats{
		ReadsCompleted:   v[0],
		ReadsMerged:      v[1],
		SectorsRead:      v[2],
		ReadTimeMs:       v[3],
		WritesCompleted:  v[4],
		WritesMerged:     v[5],
		SectorsWritten:   v[6],
		WriteTimeMs:      v[7],
		InFlight:         v[8],
		IOTimeMs:         v[9],
		WeightedIOTimeMs: v[10],
	}
	if len(v) >= 15 {
		s.HasDiscard = true
		s.DiscardsCompleted, s.SectorsDiscarded, s.DiscardTimeMs = v[11], v[13], v[14]
	}
	if len(v) >= 17 {
		s.HasFlush = true
		s.FlushesCompleted, s.FlushTimeMs = v[15], v[16]
	}
	return s, nil
}

// Read returns the current counters for d.
func Read(d Device) (Stats, error) {
	f, err := os.Open("/proc/diskstats")
	if err != nil {
		return Stats{}, err
	}
	defer f.Close()
	s, _, err := Parse(f, d.Major, d.Minor)
	return s, err
}

// ProcIO is /proc/<pid>/io. The char counts are what the process asked the
// kernel to read or write; the byte counts are what reached the storage layer.
type ProcIO struct {
	RChar, WChar uint64 // bytes passed to read()/write() and friends
	SyscR, SyscW uint64 // number of read and write syscalls

	// ReadBytes and WriteBytes are bytes this process caused to be fetched
	// from or sent to storage. WriteBytes is charged when a page cache page is
	// dirtied, not when writeback happens, so it counts each page once per
	// time it goes from clean to dirty.
	ReadBytes  uint64
	WriteBytes uint64

	// CancelledWriteBytes is dirty data that never reached the disk because
	// the file was truncated or deleted first — a log segment removed before
	// writeback got to it, for instance.
	CancelledWriteBytes uint64
}

// ParseProcIO parses the format of /proc/<pid>/io.
func ParseProcIO(r io.Reader) (ProcIO, error) {
	var p ProcIO
	fields := map[string]*uint64{
		"rchar":                 &p.RChar,
		"wchar":                 &p.WChar,
		"syscr":                 &p.SyscR,
		"syscw":                 &p.SyscW,
		"read_bytes":            &p.ReadBytes,
		"write_bytes":           &p.WriteBytes,
		"cancelled_write_bytes": &p.CancelledWriteBytes,
	}
	seen := 0
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		dst, known := fields[strings.TrimSpace(k)]
		if !known {
			continue
		}
		n, err := strconv.ParseUint(strings.TrimSpace(v), 10, 64)
		if err != nil {
			return ProcIO{}, fmt.Errorf("diskstats: /proc/self/io %s: %w", k, err)
		}
		*dst = n
		seen++
	}
	if err := sc.Err(); err != nil {
		return ProcIO{}, err
	}
	if seen != len(fields) {
		return ProcIO{}, fmt.Errorf("diskstats: /proc/self/io has %d of %d expected fields", seen, len(fields))
	}
	return p, nil
}

// ReadSelfIO returns this process's I/O counters. It fails on kernels built
// without CONFIG_TASK_IO_ACCOUNTING, and everywhere that is not Linux.
func ReadSelfIO() (ProcIO, error) {
	f, err := os.Open("/proc/self/io")
	if err != nil {
		return ProcIO{}, err
	}
	defer f.Close()
	return ParseProcIO(f)
}
