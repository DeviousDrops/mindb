package core

import (
	"errors"
	"fmt"
	stdmath "math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func buildEngine(t *testing.T, dims, n, capacity int, seed int64) (*Engine, *rand.Rand) {
	t.Helper()
	r := rand.New(rand.NewSource(seed))
	e, err := New(dims, capacity)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("vec-%04d", i)
		payload := []byte(fmt.Sprintf(`{"i":%d}`, i))
		if err := e.Insert(id, randVec(r, dims), payload); err != nil {
			t.Fatal(err)
		}
	}
	return e, r
}

func TestSnapshotRoundTrip(t *testing.T) {
	const dims, n, k = 48, 300, 10
	e, r := buildEngine(t, dims, n, 500, 11)

	path := filepath.Join(t.TempDir(), "snap.mdb")
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}

	loaded, err := Load(path, 500)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != e.Len() {
		t.Fatalf("loaded %d vectors, want %d", loaded.Len(), e.Len())
	}
	if loaded.Dims() != dims {
		t.Fatalf("loaded dims %d, want %d", loaded.Dims(), dims)
	}

	// The real assertion: searches must return the same thing, since that is
	// the only property anyone actually depends on.
	for trial := 0; trial < 25; trial++ {
		q := randVec(r, dims)
		before, err := e.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		after, err := loaded.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(before) != len(after) {
			t.Fatalf("trial %d: %d results before, %d after", trial, len(before), len(after))
		}
		for i := range before {
			if before[i].ID != after[i].ID {
				t.Fatalf("trial %d rank %d: %q before, %q after", trial, i, before[i].ID, after[i].ID)
			}
			if before[i].Score != after[i].Score {
				t.Fatalf("trial %d rank %d: score %v before, %v after",
					trial, i, before[i].Score, after[i].Score)
			}
			if string(before[i].Payload) != string(after[i].Payload) {
				t.Fatalf("trial %d rank %d: payload %q before, %q after",
					trial, i, before[i].Payload, after[i].Payload)
			}
		}
	}
}

func TestSnapshotSkipsDeleted(t *testing.T) {
	e, _ := buildEngine(t, 16, 50, 100, 12)
	for i := 0; i < 20; i++ {
		_, _ = e.Delete(fmt.Sprintf("vec-%04d", i))
	}

	path := filepath.Join(t.TempDir(), "snap.mdb")
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 30 {
		t.Fatalf("loaded %d vectors, want 30 (deleted ones must not be written)", loaded.Len())
	}

	res, err := loaded.Search(randVec(rand.New(rand.NewSource(1)), 16), 30)
	if err != nil {
		t.Fatal(err)
	}
	for _, hit := range res {
		var i int
		fmt.Sscanf(hit.ID, "vec-%04d", &i)
		if i < 20 {
			t.Fatalf("deleted vector %q survived the snapshot", hit.ID)
		}
	}
}

func TestSnapshotEmptyEngine(t *testing.T) {
	e, err := New(8, 10)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "empty.mdb")
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path, 10)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 0 {
		t.Fatalf("loaded %d vectors from an empty snapshot", loaded.Len())
	}
}

func TestSnapshotRejectsCorruption(t *testing.T) {
	e, _ := buildEngine(t, 16, 40, 100, 13)
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.mdb")
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	good, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// A flipped bit anywhere in the payload region must be caught. Without the
	// checksum this loads as plausible garbage.
	t.Run("flipped byte", func(t *testing.T) {
		bad := append([]byte(nil), good...)
		mid := len(bad) / 2
		bad[mid] ^= 0xFF
		p := filepath.Join(t.TempDir(), "bad.mdb")
		if err := os.WriteFile(p, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(p, 100); err == nil {
			t.Fatal("corrupt snapshot loaded without error")
		}
	})

	t.Run("truncated", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "trunc.mdb")
		if err := os.WriteFile(p, good[:len(good)-16], 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(p, 100); err == nil {
			t.Fatal("truncated snapshot loaded without error")
		}
	})

	t.Run("bad magic", func(t *testing.T) {
		bad := append([]byte(nil), good...)
		bad[0] = 'X'
		p := filepath.Join(t.TempDir(), "magic.mdb")
		if err := os.WriteFile(p, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(p, 100); !errors.Is(err, ErrBadMagic) {
			t.Fatalf("got %v, want ErrBadMagic", err)
		}
	})

	t.Run("bad version", func(t *testing.T) {
		bad := append([]byte(nil), good...)
		bad[8] = 99
		p := filepath.Join(t.TempDir(), "version.mdb")
		if err := os.WriteFile(p, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(p, 100); !errors.Is(err, ErrBadVersion) {
			t.Fatalf("got %v, want ErrBadVersion", err)
		}
	})
}

func TestLoadRejectsOversizedSnapshot(t *testing.T) {
	e, _ := buildEngine(t, 8, 50, 100, 14)
	path := filepath.Join(t.TempDir(), "snap.mdb")
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path, 10); !errors.Is(err, ErrSnapshotSize) {
		t.Fatalf("got %v, want ErrSnapshotSize", err)
	}
}

func TestSaveLeavesNoTempFiles(t *testing.T) {
	e, _ := buildEngine(t, 8, 20, 50, 15)
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.mdb")

	for i := 0; i < 3; i++ {
		if err := e.Save(path); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "snap.mdb" {
		names := make([]string, len(entries))
		for i, en := range entries {
			names[i] = en.Name()
		}
		t.Fatalf("snapshot dir contains %v, want just snap.mdb", names)
	}
}

func TestSaveOverwritesPreviousSnapshot(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "snap.mdb")

	first, _ := buildEngine(t, 8, 10, 50, 16)
	if err := first.Save(path); err != nil {
		t.Fatal(err)
	}

	second, _ := buildEngine(t, 8, 25, 50, 17)
	if err := second.Save(path); err != nil {
		t.Fatal(err)
	}

	loaded, err := Load(path, 50)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 25 {
		t.Fatalf("loaded %d vectors, want 25 from the second snapshot", loaded.Len())
	}
}

func TestSnapshotPreservesNormalization(t *testing.T) {
	e, _ := buildEngine(t, 32, 20, 50, 18)
	path := filepath.Join(t.TempDir(), "snap.mdb")
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path, 50)
	if err != nil {
		t.Fatal(err)
	}

	// Every stored vector must still be unit length: the cascade's error bound
	// assumes it, so drift here would quietly weaken a correctness guarantee.
	loaded.mu.RLock()
	defer loaded.mu.RUnlock()
	for slot := 0; slot < int(loaded.highWater); slot++ {
		if !loaded.live[slot] {
			continue
		}
		base := slot * loaded.dims
		var sum float64
		for _, v := range loaded.vectors[base : base+loaded.dims] {
			sum += float64(v) * float64(v)
		}
		if n := stdmath.Sqrt(sum); stdmath.Abs(n-1) > 1e-6 {
			t.Fatalf("slot %d has norm %v after reload, want 1", slot, n)
		}
	}
}
