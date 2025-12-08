# know

The "Docker" for RAG - a CLI tool for your personal knowledge base.

`know` makes it easy to ingest documents and query them using RAG (Retrieval-Augmented Generation). Services start automatically when needed.

## Features

- **Single binary CLI** - Simple verbs: `run`, `ingest`, `serve`
- **Auto-start services** - Qdrant and Docling start automatically when needed
- **Docker Model Runner default** - Uses Docker's built-in model runner
- **Document parsing with Docling** - Supports PDF, DOCX, PPTX, HTML, Markdown, and more
- **Vector storage with Qdrant** - Production-ready vector database
- **Push/Pull knowledge bases** - Share your indexed documents via Docker Hub
- **OpenAI-compatible API** - Serve your knowledge base as an API endpoint

## Quick Start

```bash
# Ingest your documents (services start automatically)
know ingest ./my-docs

# Query your knowledge base
know run "What is our refund policy?"

# Or serve as an API
know serve --port 8080
```

## Installation

### From Source

```bash
git clone https://github.com/ecurtin/know
cd know
cargo build --release
# Binary is at ./target/release/know
```

### Prerequisites

- Docker (with Docker Compose)
- One of the following LLM backends:
  - Docker Model Runner (default): Enable in Docker Desktop settings
  - OpenAI API key: `export OPENAI_API_KEY=sk-...`

## Commands

### `know run <question>`

Query your knowledge base. Services start automatically if not running.

```bash
know run "What is the deployment process?"
know run "How do I configure authentication?"
```

### `know ingest <path>`

Ingest files from a directory into the knowledge base. Services start automatically.

```bash
# Ingest all supported files
know ingest ./docs

# Ingest specific file types
know ingest ./docs --extensions md,txt,pdf,docx,html

# Use a specific collection name
know ingest ./docs --collection my-project
```

### `know serve`

Serve an OpenAI-compatible API endpoint. Services start automatically.

```bash
know serve --port 8080
```

Then query it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is our refund policy?"}]}'
```

### `know push <image>`

Push your knowledge base to Docker Hub for sharing.

```bash
know push myuser/company-docs:v1
```

### `know pull <image>`

Pull a shared knowledge base from Docker Hub.

```bash
know pull myuser/company-docs:v1
```

### `know down`

Stop the qdrant and docling services.

```bash
know down
```

### `know clean [collection]`

Clear the knowledge base.

```bash
know clean           # Clear the default 'know' collection
know clean my-project  # Clear a specific collection
```

### `know status`

Show the status of services.

```bash
know status
```

## Backend Configuration

`know` automatically detects available backends in this order:

1. **Docker Model Runner** (default) - Uses Docker's built-in AI model runner
3. **OpenAI** - Falls back if neither is available and `OPENAI_API_KEY` is set

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KNOW_BACKEND` | Force a specific backend (`docker`, `openai`) | auto-detect |
| `KNOW_BASE_URL` | Base URL for the LLM backend | varies by backend |
| `KNOW_MODEL` | Model name for text generation | varies by backend |
| `KNOW_EMBED_MODEL` | Model name for embeddings | varies by backend |
| `KNOW_QDRANT_URL` | Qdrant URL | `http://localhost:6333` |
| `KNOW_DOCLING_URL` | Docling URL | `http://localhost:5001` |
| `KNOW_COLLECTION` | Default collection name | `know` |
| `OPENAI_API_KEY` | OpenAI API key (for OpenAI backend) | - |

### Command-line Options

```bash
# Use a specific backend
know --backend docker run "What is the refund policy?"

# Use a custom endpoint (e.g., LocalAI, vLLM)
know --backend openai --base-url http://localhost:8000/v1 run "What is the refund policy?"
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   know CLI  │────▶│   Docling   │────▶│   Qdrant    │
│             │     │  (parsing)  │     │  (vectors)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│       LLM Backend (auto-detected)        │
├──────────────────┬───────────────────────┤
│      Docker      │   OpenAI-compatible   │
│   Model Runner   │       endpoint        │
└──────────────────┴───────────────────────┘
```

## Docker Compose Services

The following services are managed automatically:

- **Qdrant**: Vector database for storing embeddings
  - Port 6333 (HTTP API)
  - Port 6334 (gRPC)
  - Data persisted in `know-qdrant-data` volume

- **Docling**: Document parsing service
  - Port 5001
  - Converts PDF, DOCX, PPTX, HTML to text

