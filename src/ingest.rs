// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::path::Path;
use text_splitter::TextSplitter;

use crate::backend::create_backend;
use crate::cli::Cli;
use crate::qdrant::{DocumentChunk, QdrantClient};

const CHUNK_SIZE: usize = 512; // characters

/// Parse a document using docling service
async fn parse_with_docling(docling_url: &str, file_path: &Path) -> Result<String> {
    let client = reqwest::Client::new();

    // Read file content
    let file_content = tokio::fs::read(file_path).await?;
    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("document");

    // Create multipart form
    let part = reqwest::multipart::Part::bytes(file_content)
        .file_name(file_name.to_string())
        .mime_str("application/octet-stream")?;

    let form = reqwest::multipart::Form::new().part("files", part);

    #[derive(Deserialize)]
    struct DoclingResponse {
        document: DoclingDocument,
    }

    #[derive(Deserialize)]
    struct DoclingDocument {
        md_content: String,
    }

    let response = client
        .post(format!("{}/v1/convert/file", docling_url))
        .multipart(form)
        .send()
        .await
        .context("Failed to connect to docling")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Docling returned error {}: {}", status, text);
    }

    let result: DoclingResponse = response.json().await.context("Failed to parse docling response")?;

    Ok(result.document.md_content)
}

/// Check if docling service is available
async fn is_docling_available(docling_url: &str) -> bool {
    reqwest::get(format!("{}/health", docling_url))
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Read file content directly (fallback when docling is not available)
async fn read_file_directly(file_path: &Path) -> Result<String> {
    let content = tokio::fs::read_to_string(file_path)
        .await
        .context("Failed to read file")?;
    Ok(content)
}

/// Split text into chunks
fn chunk_text(text: &str) -> Vec<String> {
    let splitter = TextSplitter::new(CHUNK_SIZE);

    // Use character-based chunking as a simple approach
    // that respects semantic boundaries
    splitter
        .chunks(text)
        .map(|s| s.to_string())
        .filter(|s| !s.trim().is_empty())
        .collect()
}

/// Ingest documents from a path
pub async fn ingest(cli: &Cli, path: &str, extensions: &str) -> Result<()> {
    // Check if qdrant is available
    let qdrant = QdrantClient::new(Some(&cli.qdrant_url));
    if !qdrant.is_available().await {
        anyhow::bail!(
            "Qdrant is not available at {}. Run 'know up' to start services.",
            cli.qdrant_url
        );
    }

    // Create backend for embeddings
    let backend = create_backend(cli).await?;

    // Check docling availability
    let use_docling = is_docling_available(&cli.docling_url).await;
    if !use_docling {
        eprintln!(
            "Warning: Docling not available at {}. Using direct file reading (limited format support).",
            cli.docling_url
        );
    }

    // Collect files to process
    let exts: Vec<&str> = extensions.split(',').map(|s| s.trim()).collect();
    let mut files: Vec<std::path::PathBuf> = Vec::new();

    let path_obj = Path::new(path);
    if path_obj.is_file() {
        files.push(path_obj.to_path_buf());
    } else {
        let pattern = format!("{}/**/*", path);
        for entry in glob(&pattern).context("Invalid glob pattern")? {
            if let Ok(path_buf) = entry {
                if path_buf.is_file() {
                    let ext = path_buf
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    if exts.contains(&ext) {
                        files.push(path_buf);
                    }
                }
            }
        }
    }

    if files.is_empty() {
        println!("No files found matching extensions: {}", extensions);
        return Ok(());
    }

    println!("Found {} files to process", files.len());

    // Get embedding dimension from a test embedding
    let test_embedding = backend.embed("test").await?;
    let vector_size = test_embedding.len();

    // Ensure collection exists
    qdrant.ensure_collection(&cli.collection, vector_size).await?;

    // Process files with progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("Invalid progress template")
            .progress_chars("#>-"),
    );

    let mut total_chunks = 0;

    for file_path in files {
        pb.set_message(format!("Processing {}", file_path.display()));

        // Parse document
        let content = if use_docling {
            let ext = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if ["pdf", "docx", "pptx", "xlsx", "html"].contains(&ext) {
                match parse_with_docling(&cli.docling_url, &file_path).await {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Warning: Failed to parse {} with docling: {}", file_path.display(), e);
                        read_file_directly(&file_path).await.unwrap_or_default()
                    }
                }
            } else {
                read_file_directly(&file_path).await.unwrap_or_default()
            }
        } else {
            read_file_directly(&file_path).await.unwrap_or_default()
        };

        if content.is_empty() {
            pb.inc(1);
            continue;
        }

        // Chunk the content
        let chunks = chunk_text(&content);

        // Create document chunks and embeddings
        let mut doc_chunks = Vec::new();
        let mut embeddings = Vec::new();

        for chunk_content in chunks {
            let chunk = DocumentChunk {
                id: uuid::Uuid::new_v4().to_string(),
                content: chunk_content.clone(),
                source: file_path.to_string_lossy().to_string(),
            };

            match backend.embed(&chunk_content).await {
                Ok(embedding) => {
                    doc_chunks.push(chunk);
                    embeddings.push(embedding);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to embed chunk: {}", e);
                }
            }
        }

        // Batch upsert
        if !doc_chunks.is_empty() {
            qdrant
                .upsert_batch(&cli.collection, &doc_chunks, embeddings)
                .await?;
            total_chunks += doc_chunks.len();
        }

        pb.inc(1);
    }

    pb.finish_with_message("Done!");
    println!("\nIngested {} chunks into collection '{}'", total_chunks, cli.collection);

    Ok(())
}

/// Ask a question against the knowledge base
pub async fn ask(cli: &Cli, question: &str) -> Result<()> {
    // Check if qdrant is available
    let qdrant = QdrantClient::new(Some(&cli.qdrant_url));
    if !qdrant.is_available().await {
        anyhow::bail!(
            "Qdrant is not available at {}. Run 'know up' to start services.",
            cli.qdrant_url
        );
    }

    // Check if collection has data
    let info = qdrant.collection_info(&cli.collection).await?;
    if info.is_none() || info.as_ref().map(|i| i.points_count).unwrap_or(0) == 0 {
        println!("Knowledge base is empty. Run 'know ingest <path>' first.");
        return Ok(());
    }

    // Create backend
    let backend = create_backend(cli).await?;

    println!("Thinking...\n");

    // Embed the question
    let query_embedding = backend.embed(question).await?;

    // Search for relevant chunks
    let results = qdrant.search(&cli.collection, query_embedding, 5).await?;

    if results.is_empty() {
        println!("No relevant documents found.");
        return Ok(());
    }

    // Build context from search results
    let context: String = results
        .iter()
        .map(|chunk| {
            format!(
                "[Source: {}]\n{}\n",
                chunk.source,
                chunk.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n---\n");

    // Generate response
    let response = backend.generate(question, &context).await?;

    println!("{}\n", response);

    // Print sources
    println!("Sources:");
    let mut seen_sources = std::collections::HashSet::new();
    for chunk in &results {
        if seen_sources.insert(&chunk.source) {
            println!("  - {}", chunk.source);
        }
    }

    Ok(())
}
