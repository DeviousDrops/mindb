.PHONY: all build gen flatc-image check-gen asm check-cross test image image-multi bench

# Generated code is committed, so a plain build needs no toolchain beyond Go.
all: build

build:
	go build -o bin/mindb-server cmd/mindb-server/main.go

# flatc must match the FlatBuffers Go runtime in go.mod. Generated code and
# runtime are a matched pair, and a mismatch does not fail the build: it shifts
# vtable offsets and surfaces as garbage fields at runtime. Hence a pinned
# container rather than whatever flatc happens to be on PATH.
FLATC_VERSION := 25.12.19
FLATC_IMAGE   := mindb-flatc:$(FLATC_VERSION)

flatc-image:
	docker image inspect $(FLATC_IMAGE) >/dev/null 2>&1 || \
		docker build -f build/flatc.Dockerfile -t $(FLATC_IMAGE) \
			--build-arg FLATBUFFERS_VERSION=v$(FLATC_VERSION) .

gen: flatc-image
	docker run --rm -v "$(CURDIR)":/w -w /w $(FLATC_IMAGE) --go --grpc -o pkg/ fbs/mindb.fbs

# Fails when the committed generated code does not match the schema, which is
# the only thing stopping the two from drifting apart unnoticed.
check-gen: gen
	git diff --exit-code -- pkg/mindb

STUB := pkg/math/dotint8_avx2_stub_amd64.go

# Regenerates the AVX2 kernel and its stub, then re-adds the build tag avo does
# not emit. Only needed when pkg/math/avo/asm.go changes.
asm:
	cd pkg/math && go run avo/asm.go -out dotint8_avx2_amd64.s -stubs dotint8_avx2_stub_amd64.go -pkg math
	printf '//go:build amd64\n\n' > $(STUB).tmp && cat $(STUB) >> $(STUB).tmp && mv $(STUB).tmp $(STUB)
	gofmt -w $(STUB)

# The deploy target is arm64, which nothing here builds for by default. A
# missing build tag on an amd64-only file only fails under a cross-build, so
# this covers every architecture the project claims to support.
check-cross:
	GOOS=linux GOARCH=amd64 go vet ./...
	GOOS=linux GOARCH=arm64 go vet ./...
	GOOS=linux GOARCH=riscv64 go vet ./...

test:
	go test ./...

IMAGE   := mindb
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

# A single-architecture image for the machine you are on. VERSION is stamped
# into the binary and reported in the first line of the startup log, so an
# image running somewhere you cannot reach can still say what it is.
image:
	docker build -f build/Dockerfile --build-arg VERSION=$(VERSION) -t $(IMAGE):$(VERSION) .

# What CI publishes. Needs a buildx builder that can export a manifest list;
# on the plain docker driver this fails unless the containerd image store is
# enabled. Nothing is pushed -- this only proves the Dockerfile works for both.
image-multi:
	docker buildx build -f build/Dockerfile --platform linux/amd64,linux/arm64 \
		--build-arg VERSION=$(VERSION) -t $(IMAGE):$(VERSION) .

# The benchmarks quoted in README.md. They are reported per architecture
# because the cascade is on with AVX2 and off without it, so the machine is
# part of the number -- hence the banner.
#
# Never run this under emulation. It would time QEMU.
bench:
	@go env GOARCH GOOS | tr '\n' ' '
	@echo ""
	go test ./pkg/math -run '^$$' -bench 'BenchmarkDot|BenchmarkDotInt8' -benchmem
	go test ./pkg/core -run '^$$' -bench 'BenchmarkSearchPaths|BenchmarkWALInsert' -benchmem
