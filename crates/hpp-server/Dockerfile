# Multi-stage Docker build for ultra-small, ultra-fast Rust microservice
FROM rust:1.80-slim as builder

WORKDIR /usr/src/app

# Copy workspace and manifests
COPY Cargo.toml ./
COPY crates/hpp-core ./crates/hpp-core
COPY crates/hpp-server ./crates/hpp-server
COPY models/weights.json ./models/weights.json

# Build static release binary
WORKDIR /usr/src/app/crates/hpp-server
RUN cargo build --release

# Final runtime image: scratch/debian-slim minimal image (<30MB)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/app/target/release/hpp-server /app/hpp-server
COPY static /app/static

ENV PORT=8000
EXPOSE 8000

CMD ["/app/hpp-server"]
