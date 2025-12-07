// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub source: String,
}

#[derive(Serialize, Debug)]
struct CreateCollectionRequest {
    vectors: VectorConfig,
}

#[derive(Serialize, Debug)]
struct VectorConfig {
    size: usize,
    distance: String,
}

#[derive(Serialize, Debug)]
struct UpsertPointsRequest {
    points: Vec<Point>,
}

#[derive(Serialize, Debug)]
struct Point {
    id: String,
    vector: Vec<f32>,
    payload: PointPayload,
}

#[derive(Serialize, Deserialize, Debug)]
struct PointPayload {
    content: String,
    source: String,
}

#[derive(Serialize, Debug)]
struct SearchRequest {
    vector: Vec<f32>,
    limit: usize,
    with_payload: bool,
}

#[derive(Deserialize, Debug)]
struct SearchResponse {
    result: Vec<SearchResult>,
}

#[derive(Deserialize, Debug)]
struct SearchResult {
    #[allow(dead_code)]
    id: serde_json::Value,
    #[allow(dead_code)]
    score: f32,
    payload: Option<PointPayload>,
}

pub struct QdrantClient {
    client: reqwest::Client,
    base_url: String,
}

impl QdrantClient {
    pub fn new(base_url: Option<&str>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.unwrap_or(DEFAULT_QDRANT_URL).to_string(),
        }
    }

    /// Ensure a collection exists with the right vector size
    pub async fn ensure_collection(&self, collection: &str, vector_size: usize) -> Result<()> {
        // Check if collection exists
        let response = self
            .client
            .get(format!("{}/collections/{}", self.base_url, collection))
            .send()
            .await;

        if let Ok(resp) = response {
            if resp.status().is_success() {
                return Ok(());
            }
        }

        // Create collection
        let request = CreateCollectionRequest {
            vectors: VectorConfig {
                size: vector_size,
                distance: "Cosine".to_string(),
            },
        };

        self.client
            .put(format!("{}/collections/{}", self.base_url, collection))
            .json(&request)
            .send()
            .await
            .context("Failed to create collection")?
            .error_for_status()
            .context("Failed to create collection")?;

        Ok(())
    }

    /// Store a document chunk with its embedding
    #[allow(dead_code)]
    pub async fn upsert(
        &self,
        collection: &str,
        chunk: &DocumentChunk,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let request = UpsertPointsRequest {
            points: vec![Point {
                id: chunk.id.clone(),
                vector: embedding,
                payload: PointPayload {
                    content: chunk.content.clone(),
                    source: chunk.source.clone(),
                },
            }],
        };

        self.client
            .put(format!("{}/collections/{}/points", self.base_url, collection))
            .json(&request)
            .send()
            .await
            .context("Failed to upsert point")?
            .error_for_status()
            .context("Failed to upsert point")?;

        Ok(())
    }

    /// Batch upsert multiple chunks
    pub async fn upsert_batch(
        &self,
        collection: &str,
        chunks: &[DocumentChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        let points: Vec<Point> = chunks
            .iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| Point {
                id: chunk.id.clone(),
                vector: embedding,
                payload: PointPayload {
                    content: chunk.content.clone(),
                    source: chunk.source.clone(),
                },
            })
            .collect();

        let request = UpsertPointsRequest { points };

        self.client
            .put(format!("{}/collections/{}/points", self.base_url, collection))
            .json(&request)
            .send()
            .await
            .context("Failed to batch upsert points")?
            .error_for_status()
            .context("Failed to batch upsert points")?;

        Ok(())
    }

    /// Search for similar documents
    pub async fn search(
        &self,
        collection: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<DocumentChunk>> {
        let request = SearchRequest {
            vector: query_embedding,
            limit,
            with_payload: true,
        };

        let response: SearchResponse = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, collection
            ))
            .json(&request)
            .send()
            .await
            .context("Failed to search")?
            .json()
            .await
            .context("Failed to parse search response")?;

        let chunks = response
            .result
            .into_iter()
            .filter_map(|r| {
                r.payload.map(|p| DocumentChunk {
                    id: String::new(), // ID not needed for search results
                    content: p.content,
                    source: p.source,
                })
            })
            .collect();

        Ok(chunks)
    }

    /// Get collection info
    pub async fn collection_info(&self, collection: &str) -> Result<Option<CollectionInfo>> {
        let response = self
            .client
            .get(format!("{}/collections/{}", self.base_url, collection))
            .send()
            .await
            .context("Failed to get collection info")?;

        if !response.status().is_success() {
            return Ok(None);
        }

        #[derive(Deserialize)]
        struct InfoResponse {
            result: CollectionResult,
        }

        #[derive(Deserialize)]
        struct CollectionResult {
            points_count: usize,
            indexed_vectors_count: usize,
        }

        let info: InfoResponse = response.json().await?;

        Ok(Some(CollectionInfo {
            points_count: info.result.points_count,
            indexed_vectors_count: info.result.indexed_vectors_count,
        }))
    }

    /// Delete a collection
    pub async fn delete_collection(&self, collection: &str) -> Result<()> {
        self.client
            .delete(format!("{}/collections/{}", self.base_url, collection))
            .send()
            .await
            .context("Failed to delete collection")?;

        Ok(())
    }

    /// Check if qdrant is available
    pub async fn is_available(&self) -> bool {
        self.client
            .get(format!("{}/readyz", self.base_url))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

pub struct CollectionInfo {
    pub points_count: usize,
    #[allow(dead_code)]
    pub indexed_vectors_count: usize,
}

/// Clean/delete a collection
pub async fn clean(collection: &str) -> Result<()> {
    let client = QdrantClient::new(None);

    if !client.is_available().await {
        println!("Qdrant is not available. Run 'know up' first.");
        return Ok(());
    }

    client.delete_collection(collection).await?;
    println!("Collection '{}' deleted.", collection);

    Ok(())
}
