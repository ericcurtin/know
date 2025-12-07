// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use crate::backend::{create_backend, LlmBackend};
use crate::cli::Cli;
use crate::qdrant::QdrantClient;

struct AppState {
    backend: Box<dyn LlmBackend>,
    qdrant: QdrantClient,
    collection: String,
}

// OpenAI-compatible request/response types
#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    #[allow(dead_code)]
    stream: bool,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize {
    5
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}


#[derive(Serialize)]
struct HealthResponse {
    status: String,
    backend: String,
    collection: String,
}

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        backend: state.backend.name().to_string(),
        collection: state.collection.clone(),
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Extract the user's question from the last message
    let user_message = request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    if user_message.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "No user message found",
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response();
    }

    // Embed the question
    let query_embedding = match state.backend.embed(&user_message).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to embed query: {}", e),
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Search for relevant chunks
    let results = match state
        .qdrant
        .search(&state.collection, query_embedding, request.top_k)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to search: {}", e),
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Build context
    let context: String = results
        .iter()
        .map(|chunk| format!("[Source: {}]\n{}", chunk.source, chunk.content))
        .collect::<Vec<_>>()
        .join("\n---\n");

    // Generate response
    let response = match state.backend.generate(&user_message, &context).await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to generate response: {}", e),
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Build OpenAI-compatible response
    let completion = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: "know-rag".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    Json(completion).into_response()
}

/// Serve an OpenAI-compatible API endpoint
pub async fn serve(cli: &Cli, port: u16) -> Result<()> {
    // Check if qdrant is available
    let qdrant = QdrantClient::new(Some(&cli.qdrant_url));
    if !qdrant.is_available().await {
        anyhow::bail!(
            "Qdrant is not available at {}. Run 'know up' to start services.",
            cli.qdrant_url
        );
    }

    // Create backend
    let backend = create_backend(cli).await?;

    println!("Using backend: {}", backend.name());

    let state = Arc::new(AppState {
        backend,
        qdrant,
        collection: cli.collection.clone(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(cors)
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("Starting know server on http://{}", addr);
    println!("\nOpenAI-compatible endpoint: http://localhost:{}/v1/chat/completions", port);
    println!("Health check: http://localhost:{}/health", port);
    println!("\nExample usage:");
    println!("  curl http://localhost:{}/v1/chat/completions \\", port);
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"messages\": [{{\"role\": \"user\", \"content\": \"What is our refund policy?\"}}]}}'");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
