// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::process::Command;

/// Get the path to the docker-compose.yml file
fn get_compose_file() -> Result<String> {
    // First check if there's a docker-compose.yml in the current directory
    if std::path::Path::new("docker-compose.yml").exists() {
        return Ok("docker-compose.yml".to_string());
    }

    // Then check the executable directory
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let compose_path = exe_dir.join("docker-compose.yml");
            if compose_path.exists() {
                return Ok(compose_path.to_string_lossy().to_string());
            }
        }
    }

    // Fall back to embedded compose file - write it to a temp location
    let temp_dir = std::env::temp_dir().join("know");
    std::fs::create_dir_all(&temp_dir)?;
    let compose_path = temp_dir.join("docker-compose.yml");

    let compose_content = include_str!("../docker-compose.yml");
    std::fs::write(&compose_path, compose_content)?;

    Ok(compose_path.to_string_lossy().to_string())
}

/// Check if qdrant is running and healthy
async fn is_qdrant_ready() -> bool {
    reqwest::get("http://localhost:6333/readyz")
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Ensure services are running, starting them if necessary
pub async fn ensure_running() -> Result<()> {
    // Quick check if qdrant is already running
    if is_qdrant_ready().await {
        return Ok(());
    }

    // Start services
    let compose_file = get_compose_file()?;
    eprintln!("Starting services...");

    let status = Command::new("docker")
        .args(["compose", "-f", &compose_file, "up", "-d"])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("Failed to run docker compose. Is Docker running?")?;

    if !status.success() {
        anyhow::bail!("Failed to start services. Run 'docker compose up' manually to see errors.");
    }

    // Wait for qdrant to be ready (up to 30 seconds)
    eprintln!("Waiting for services to be ready...");
    for i in 0..30 {
        if is_qdrant_ready().await {
            eprintln!("Services ready.");
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        if i == 10 {
            eprintln!("Still waiting for qdrant...");
        }
    }

    anyhow::bail!("Services started but qdrant not ready after 30s. Run 'know status' to check.")
}

/// Stop qdrant and docling services
pub async fn down() -> Result<()> {
    let compose_file = get_compose_file()?;
    println!("Stopping know services...");

    let status = Command::new("docker")
        .args(["compose", "-f", &compose_file, "down"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("Failed to run docker compose")?;

    if !status.success() {
        anyhow::bail!("docker compose down failed");
    }

    println!("Services stopped.");

    Ok(())
}

/// Show status of services
pub async fn status() -> Result<()> {
    let compose_file = get_compose_file()?;

    println!("Service Status:\n");

    let output = Command::new("docker")
        .args(["compose", "-f", &compose_file, "ps", "--format", "table"])
        .output()
        .await
        .context("Failed to run docker compose ps")?;

    if output.stdout.is_empty() {
        println!("No services running. Services start automatically with 'know run' or 'know ingest'.");
    } else {
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }

    // Check individual service health
    println!("\nHealth Checks:");

    // Check Qdrant
    let qdrant_status = reqwest::get("http://localhost:6333/readyz")
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    println!(
        "  Qdrant:  {}",
        if qdrant_status { "healthy" } else { "not available" }
    );

    // Check Docling
    let docling_status = reqwest::get("http://localhost:5001/health")
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    println!(
        "  Docling: {}",
        if docling_status { "healthy" } else { "not available" }
    );

    Ok(())
}
