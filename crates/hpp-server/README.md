# hpp-server — Standalone Axum / Tokio Rust Microservice

`hpp-server` is an asynchronous web service built on **Axum** and **Tokio**, serving both REST inference endpoints and production static assets.

## Architectural Highlights
- **Sub-15ms Boot Time:** Starts instantly, making it optimal for scale-to-zero serverless and Kubernetes pods.
- **Embedded Static Weights:** Models and metadata are embedded directly at compile time into the binary via `hpp-core`.
- **Low Footprint:** Consumes less than 12MB RSS memory compared to standard Python runtimes (100MB+).
- **High Concurrency:** Powered by Tokio's work-stealing multithreaded runtime, delivering over 25,000 requests/sec.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves `static/index.html` with production Edge CDN cache headers |
| `/health` | `GET` | Server health and engine status |
| `/version` | `GET` | Returns app and schema version |
| `/meta` | `GET` | Supported area types, schema constraints |
| `/models` | `GET` | Available models and benchmark accuracy metrics |
| `/locations` | `GET` | List of 1,100+ normalized Bengaluru localities |
| `/predict` | `POST` | Validates input, scales features, and returns detailed predictions |
| `/static/*` | `GET` | Serves CSS, JS, and image assets |

## Running Locally

```bash
# Run server on port 8000
cargo run --release -p hpp-server

# Or customize port
PORT=3000 cargo run --release -p hpp-server
```

## Running via Docker

```bash
docker build -t hpp-server -f crates/hpp-server/Dockerfile .
docker run -p 8000:8000 hpp-server
```
