package diskstats

import (
	"runtime"
	"strings"
	"testing"
)

// Three generations of the format: 11 fields (pre-4.18), 15 with discards
// (4.18), 17 with flushes (5.5+). A parser that only knew the newest would
// fail on the long-term-support kernels still in production.
const sample = `   8       0 sda 100 1 2000 30 400 5 8000 90 0 120 140
   8       1 sda1 10 0 200 3 40 0 800 9 0 12 14 7 0 56 2
 259       0 nvme0n1 11302 4810 665258 4097 33443 47431 4023368 12285 2 8884 20057 17665 0 2068152 2816 7864 857
 253       0 dm-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
`

func TestParseGenerations(t *testing.T) {
	cases := []struct {
		major, minor uint32
		name         string
		check        func(t *testing.T, s Stats)
	}{
		{8, 0, "sda", func(t *testing.T, s Stats) {
			if s.WritesCompleted != 400 || s.SectorsWritten != 8000 || s.WriteTimeMs != 90 || s.WeightedIOTimeMs != 140 {
				t.Errorf("%+v", s)
			}
			if s.HasDiscard || s.HasFlush {
				t.Error("11-field line claims discard or flush fields")
			}
		}},
		{8, 1, "sda1", func(t *testing.T, s Stats) {
			if !s.HasDiscard || s.HasFlush || s.DiscardsCompleted != 7 || s.SectorsDiscarded != 56 || s.DiscardTimeMs != 2 {
				t.Errorf("%+v", s)
			}
		}},
		{259, 0, "nvme0n1", func(t *testing.T, s Stats) {
			if !s.HasFlush || s.FlushesCompleted != 7864 || s.FlushTimeMs != 857 || s.InFlight != 2 {
				t.Errorf("%+v", s)
			}
			if s.ReadsCompleted != 11302 || s.SectorsWritten != 4023368 || s.IOTimeMs != 8884 {
				t.Errorf("%+v", s)
			}
		}},
		// Newer kernels may append fields; extra ones are ignored, not an error.
		{253, 0, "dm-0", func(t *testing.T, s Stats) {
			if !s.HasFlush || s.FlushesCompleted != 16 || s.FlushTimeMs != 17 {
				t.Errorf("%+v", s)
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s, name, err := Parse(strings.NewReader(sample), c.major, c.minor)
			if err != nil {
				t.Fatal(err)
			}
			if name != c.name {
				t.Fatalf("name %q, want %q", name, c.name)
			}
			c.check(t, s)
		})
	}
}

func TestParseMissingAndMalformed(t *testing.T) {
	if _, _, err := Parse(strings.NewReader(sample), 9, 9); err == nil {
		t.Error("absent device should be an error")
	}
	if _, _, err := Parse(strings.NewReader("8 0 sda 1 2 3\n"), 8, 0); err == nil {
		t.Error("short line should be an error")
	}
	if _, _, err := Parse(strings.NewReader("8 0 sda 1 2 3 4 5 6 7 8 9 10 x\n"), 8, 0); err == nil {
		t.Error("non-numeric field should be an error")
	}
}

func TestParseProcIO(t *testing.T) {
	const in = `rchar: 4092
wchar: 1048576
syscr: 9
syscw: 64
read_bytes: 0
write_bytes: 1052672
cancelled_write_bytes: 4096
`
	p, err := ParseProcIO(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := ProcIO{RChar: 4092, WChar: 1048576, SyscR: 9, SyscW: 64, WriteBytes: 1052672, CancelledWriteBytes: 4096}
	if p != want {
		t.Fatalf("got %+v\nwant %+v", p, want)
	}

	if _, err := ParseProcIO(strings.NewReader("rchar: 1\n")); err == nil {
		t.Error("a truncated file should be an error, not zeros")
	}
}

// On a real Linux host the test's own working directory is on some device.
// Whether it is a block device depends on where the test runs — a container's
// overlayfs is not — so both answers are accepted, but each must be coherent.
func TestDeviceForLive(t *testing.T) {
	devs, err := DeviceFor(t.TempDir())
	if runtime.GOOS != "linux" {
		if err != ErrUnsupported {
			t.Fatalf("want ErrUnsupported off Linux, got %v", err)
		}
		return
	}
	if err != nil {
		t.Skipf("temp dir is not on a block device here: %v", err)
	}
	for _, d := range devs {
		if _, err := Read(d); err != nil {
			t.Fatalf("Read(%s): %v", d, err)
		}
	}
	t.Logf("temp dir is on %v", devs)

	if _, err := ReadSelfIO(); err != nil {
		t.Logf("no per-process I/O accounting here: %v", err)
	}
}
