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

/// Start qdrant and docling services
pub async fn up() -> Result<()> {
    let compose_file = get_compose_file()?;
    println!("Starting know services...");

    let status = Command::new("docker")
        .args(["compose", "-f", &compose_file, "up", "-d"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("Failed to run docker compose")?;

    if !status.success() {
        anyhow::bail!("docker compose up failed");
    }

    println!("\nServices started successfully!");
    println!("  - Qdrant:  http://localhost:6333");
    println!("  - Docling: http://localhost:5001");
    println!("\nRun 'know status' to check service health.");

    Ok(())
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
        println!("No services running. Run 'know up' to start services.");
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
