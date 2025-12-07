// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "know")]
#[command(version)]
#[command(about = "The 'Ollama' for RAG - a CLI tool for your personal knowledge base")]
#[command(long_about = r#"
know - Your personal knowledge base CLI

know makes it easy to ingest documents and query them using RAG
(Retrieval-Augmented Generation). It uses docker-compose to manage
qdrant (vector database) and docling (document parsing) services.

Backend Priority (automatic fallback):
  1. Docker Model Runner (default, if available)
  2. Ollama (fallback, if available)
  3. OpenAI-compatible API (if configured)

Examples:
  # Start the services
  $ know up

  # Ingest documents
  $ know ingest ./docs --extensions md,txt,pdf

  # Ask questions
  $ know ask "What is the refund policy?"

  # Push knowledge base to Docker Hub
  $ know push myuser/company-docs:v1

  # Pull knowledge base from Docker Hub
  $ know pull myuser/company-docs:v1
"#)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Backend provider to use (auto-detects if not specified)
    #[arg(short, long, global = true, env = "KNOW_BACKEND")]
    pub backend: Option<BackendType>,

    /// Base URL for the LLM backend
    #[arg(long, global = true, env = "KNOW_BASE_URL")]
    pub base_url: Option<String>,

    /// Model name for text generation
    #[arg(long, global = true, env = "KNOW_MODEL")]
    pub model: Option<String>,

    /// Model name for embeddings
    #[arg(long, global = true, env = "KNOW_EMBED_MODEL")]
    pub embed_model: Option<String>,

    /// Qdrant URL
    #[arg(long, global = true, default_value = "http://localhost:6333", env = "KNOW_QDRANT_URL")]
    pub qdrant_url: String,

    /// Docling URL
    #[arg(long, global = true, default_value = "http://localhost:5001", env = "KNOW_DOCLING_URL")]
    pub docling_url: String,

    /// Collection name in qdrant
    #[arg(long, global = true, default_value = "know", env = "KNOW_COLLECTION")]
    pub collection: String,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start qdrant and docling services via docker-compose
    Up,

    /// Stop qdrant and docling services
    Down,

    /// Ingest files from a directory into the knowledge base
    Ingest {
        /// The path to the directory or file to ingest
        path: String,

        /// File extensions to look for (comma-separated: md,txt,pdf,docx)
        #[arg(long, default_value = "md,txt,pdf,docx,html")]
        extensions: String,
    },

    /// Ask a question based on your knowledge base
    Ask {
        /// The question you want to ask
        query: Vec<String>,
    },

    /// Serve an OpenAI-compatible API endpoint
    Serve {
        /// Port to serve on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// Clear the knowledge base (or a specific collection)
    Clean {
        /// Collection name to clean (defaults to 'know')
        #[arg(default_value = "know")]
        collection: String,
    },

    /// Push the vector database to Docker Hub
    Push {
        /// Image name (e.g., myuser/my-knowledge:v1)
        name: String,
    },

    /// Pull a vector database from Docker Hub
    Pull {
        /// Image name (e.g., myuser/my-knowledge:v1)
        name: String,
    },

    /// Show status of services
    Status,
}

#[derive(Clone, ValueEnum, Debug, PartialEq)]
pub enum BackendType {
    /// Docker Model Runner (default)
    Docker,
    /// Ollama
    Ollama,
    /// OpenAI-compatible API
    Openai,
}
