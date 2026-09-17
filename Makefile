.PHONY: all build flatc asm check-cross test

all: flatc build

flatc:
	flatc --go --grpc -o pkg/ fbs/mindb.fbs

build:
	go build -o bin/mindb-server cmd/mindb-server/main.go

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
