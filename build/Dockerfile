# MinDB server image: static binary, distroless base, non-root.
#
# The builder runs on the *build* platform and Go cross-compiles to the target.
# Emulating the whole Go toolchain under QEMU to produce an arm64 binary takes
# minutes per build and buys nothing, because CGO is off and the cross-compile
# is native-speed. That is what --platform=$BUILDPLATFORM buys here, and it is
# why TARGETARCH is read rather than inferred.
ARG GO_VERSION=1.25

FROM --platform=$BUILDPLATFORM golang:${GO_VERSION} AS build

WORKDIR /src

# Dependencies first, so a source-only change does not re-download the module
# cache on every build.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# TARGETARCH is supplied by buildx. VERSION is stamped into the binary and
# reported in the first line of the startup log.
ARG TARGETARCH
ARG VERSION=dev
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" \
    -o /out/mindb-server ./cmd/mindb-server

# distroless/static: no shell, no package manager, no libc. It works because
# CGO_ENABLED=0 above makes the binary genuinely static -- the race detector
# and anything else needing cgo cannot run in this image, which is fine, those
# belong in CI.
#
# The cost worth knowing: there is no shell, so `kubectl exec` into a wedged
# pod gets you nothing. Swap :nonroot for :debug-nonroot on the day that
# matters.
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/mindb-server /mindb-server

# 50051 is the FlatBuffers data service; 50052 is grpc.health.v1.Health, which
# is on its own listener because the data server forces a codec that would
# panic on a protobuf health response.
EXPOSE 50051 50052

# Snapshots and log segments go here. Deliberately not a VOLUME: that creates
# anonymous volumes under `docker run` and does nothing for Kubernetes. Mount
# something writable by uid 65532 and point -snapshot at it.
WORKDIR /data

# From the distroless base: uid/gid 65532.
USER nonroot:nonroot

# No CMD. Every flag is a deployment decision -- most importantly -snapshot,
# without which the server is memory-only and the write-ahead log is off.
ENTRYPOINT ["/mindb-server"]
