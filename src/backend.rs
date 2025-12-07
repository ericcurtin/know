// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::cli::{BackendType, Cli};

/// Trait for LLM backends that provide embeddings and text generation
#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn generate(&self, prompt: &str, context: &str) -> Result<String>;
    fn name(&self) -> &'static str;
}

/// Docker Model Runner backend (default)
pub struct DockerModelRunner {
    client: reqwest::Client,
    base_url: String,
    gen_model: String,
    embed_model: String,
}

impl DockerModelRunner {
    pub fn new(base_url: Option<String>, gen_model: Option<String>, embed_model: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            // Docker Model Runner uses the Docker socket via a special endpoint
            base_url: base_url.unwrap_or_else(|| "http://localhost:12434".to_string()),
            gen_model: gen_model.unwrap_or_else(|| "ai/llama3.2:3B-Q8_0".to_string()),
            embed_model: embed_model.unwrap_or_else(|| "ai/mxbai-embed-large:335M-F16".to_string()),
        }
    }

    pub async fn is_available(&self) -> bool {
        // Check if Docker Model Runner is available by testing the embeddings endpoint
        // Just checking if the server responds isn't enough - we need to verify
        // the embedding model is actually available
        #[derive(Serialize)]
        struct EmbedRequest {
            model: String,
            input: String,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedData>,
        }

        #[derive(Deserialize)]
        struct EmbedData {
            embedding: Vec<f32>,
        }

        let result = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .timeout(std::time::Duration::from_secs(10))
            .json(&EmbedRequest {
                model: self.embed_model.clone(),
                input: "test".to_string(),
            })
            .send()
            .await;

        match result {
            Ok(response) => {
                if !response.status().is_success() {
                    return false;
                }
                response.json::<EmbedResponse>().await.is_ok()
            }
            Err(_) => false,
        }
    }
}

#[async_trait]
impl LlmBackend for DockerModelRunner {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct EmbedRequest {
            model: String,
            input: String,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedData>,
        }

        #[derive(Deserialize)]
        struct EmbedData {
            embedding: Vec<f32>,
        }

        let res = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&EmbedRequest {
                model: self.embed_model.clone(),
                input: text.to_string(),
            })
            .send()
            .await?
            .json::<EmbedResponse>()
            .await
            .context("Failed to parse embedding response from Docker Model Runner")?;

        res.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("No embedding data returned")
    }

    async fn generate(&self, prompt: &str, context: &str) -> Result<String> {
        #[derive(Serialize)]
        struct ChatRequest {
            model: String,
            messages: Vec<ChatMessage>,
            stream: bool,
        }

        #[derive(Serialize)]
        struct ChatMessage {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct ChatResponse {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ResponseMessage,
        }

        #[derive(Deserialize)]
        struct ResponseMessage {
            content: String,
        }

        let system_prompt = format!(
            "You are a helpful assistant. Answer the user's question using only the context provided below. \
            If the context doesn't contain relevant information, say so.\n\nContext:\n{}",
            context
        );

        let res = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&ChatRequest {
                model: self.gen_model.clone(),
                messages: vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: system_prompt,
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: prompt.to_string(),
                    },
                ],
                stream: false,
            })
            .send()
            .await?
            .json::<ChatResponse>()
            .await
            .context("Failed to parse generation response from Docker Model Runner")?;

        res.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .context("No response generated")
    }

    fn name(&self) -> &'static str {
        "Docker Model Runner"
    }
}

/// Ollama backend (fallback)
pub struct OllamaBackend {
    client: reqwest::Client,
    base_url: String,
    gen_model: String,
    embed_model: String,
}

impl OllamaBackend {
    pub fn new(base_url: Option<String>, gen_model: Option<String>, embed_model: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            gen_model: gen_model.unwrap_or_else(|| "llama3.2".to_string()),
            embed_model: embed_model.unwrap_or_else(|| "nomic-embed-text".to_string()),
        }
    }

    pub async fn is_available(&self) -> bool {
        // Check if Ollama is available and the embedding model works
        #[derive(Serialize)]
        struct EmbedRequest {
            model: String,
            prompt: String,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            embedding: Vec<f32>,
        }

        let result = self
            .client
            .post(format!("{}/api/embeddings", self.base_url))
            .timeout(std::time::Duration::from_secs(30)) // Ollama may need to load model
            .json(&EmbedRequest {
                model: self.embed_model.clone(),
                prompt: "test".to_string(),
            })
            .send()
            .await;

        match result {
            Ok(response) => {
                if !response.status().is_success() {
                    return false;
                }
                response.json::<EmbedResponse>().await.is_ok()
            }
            Err(_) => false,
        }
    }
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct EmbedRequest {
            model: String,
            prompt: String,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            embedding: Vec<f32>,
        }

        let res = self
            .client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&EmbedRequest {
                model: self.embed_model.clone(),
                prompt: text.to_string(),
            })
            .send()
            .await?
            .json::<EmbedResponse>()
            .await
            .context("Failed to parse embedding response from Ollama")?;

        Ok(res.embedding)
    }

    async fn generate(&self, prompt: &str, context: &str) -> Result<String> {
        #[derive(Serialize)]
        struct GenerateRequest {
            model: String,
            prompt: String,
            stream: bool,
        }

        #[derive(Deserialize)]
        struct GenerateResponse {
            response: String,
        }

        let full_prompt = format!(
            "You are a helpful assistant. Answer the user's question using only the context provided below. \
            If the context doesn't contain relevant information, say so.\n\n\
            Context:\n{}\n\nQuestion: {}",
            context, prompt
        );

        let res = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&GenerateRequest {
                model: self.gen_model.clone(),
                prompt: full_prompt,
                stream: false,
            })
            .send()
            .await?
            .json::<GenerateResponse>()
            .await
            .context("Failed to parse generation response from Ollama")?;

        Ok(res.response)
    }

    fn name(&self) -> &'static str {
        "Ollama"
    }
}

/// OpenAI-compatible backend
pub struct OpenAiBackend {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    gen_model: String,
    embed_model: String,
}

impl OpenAiBackend {
    pub fn new(base_url: Option<String>, gen_model: Option<String>, embed_model: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            gen_model: gen_model.unwrap_or_else(|| "gpt-4o".to_string()),
            embed_model: embed_model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
        }
    }

    pub fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

#[async_trait]
impl LlmBackend for OpenAiBackend {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct EmbedRequest {
            model: String,
            input: String,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedData>,
        }

        #[derive(Deserialize)]
        struct EmbedData {
            embedding: Vec<f32>,
        }

        let res = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&EmbedRequest {
                model: self.embed_model.clone(),
                input: text.to_string(),
            })
            .send()
            .await?
            .json::<EmbedResponse>()
            .await
            .context("Failed to parse embedding response from OpenAI")?;

        res.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("No embedding data returned")
    }

    async fn generate(&self, prompt: &str, context: &str) -> Result<String> {
        #[derive(Serialize)]
        struct ChatRequest {
            model: String,
            messages: Vec<ChatMessage>,
        }

        #[derive(Serialize)]
        struct ChatMessage {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct ChatResponse {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ResponseMessage,
        }

        #[derive(Deserialize)]
        struct ResponseMessage {
            content: String,
        }

        let system_prompt = format!(
            "You are a helpful assistant. Answer the user's question using only the context provided below. \
            If the context doesn't contain relevant information, say so.\n\nContext:\n{}",
            context
        );

        let res = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&ChatRequest {
                model: self.gen_model.clone(),
                messages: vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: system_prompt,
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: prompt.to_string(),
                    },
                ],
            })
            .send()
            .await?
            .json::<ChatResponse>()
            .await
            .context("Failed to parse generation response from OpenAI")?;

        res.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .context("No response generated")
    }

    fn name(&self) -> &'static str {
        "OpenAI"
    }
}

/// Detect and create the best available backend
pub async fn create_backend(cli: &Cli) -> Result<Box<dyn LlmBackend>> {
    // If user explicitly specified a backend, use it
    if let Some(ref backend_type) = cli.backend {
        return match backend_type {
            BackendType::Docker => Ok(Box::new(DockerModelRunner::new(
                cli.base_url.clone(),
                cli.model.clone(),
                cli.embed_model.clone(),
            ))),
            BackendType::Ollama => Ok(Box::new(OllamaBackend::new(
                cli.base_url.clone(),
                cli.model.clone(),
                cli.embed_model.clone(),
            ))),
            BackendType::Openai => Ok(Box::new(OpenAiBackend::new(
                cli.base_url.clone(),
                cli.model.clone(),
                cli.embed_model.clone(),
            ))),
        };
    }

    // Auto-detect: try Docker Model Runner first, then Ollama, then OpenAI
    let docker_runner = DockerModelRunner::new(
        cli.base_url.clone(),
        cli.model.clone(),
        cli.embed_model.clone(),
    );
    if docker_runner.is_available().await {
        eprintln!("Using Docker Model Runner backend");
        return Ok(Box::new(docker_runner));
    }

    let ollama = OllamaBackend::new(
        cli.base_url.clone(),
        cli.model.clone(),
        cli.embed_model.clone(),
    );
    if ollama.is_available().await {
        eprintln!("Using Ollama backend");
        return Ok(Box::new(ollama));
    }

    let openai = OpenAiBackend::new(
        cli.base_url.clone(),
        cli.model.clone(),
        cli.embed_model.clone(),
    );
    if openai.is_available() {
        eprintln!("Using OpenAI backend");
        return Ok(Box::new(openai));
    }

    anyhow::bail!(
        "No LLM backend available. Please either:\n\
        1. Start Docker Model Runner: docker model serve\n\
        2. Start Ollama: ollama serve\n\
        3. Set OPENAI_API_KEY environment variable"
    )
}
