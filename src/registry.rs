// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::process::Command;

const QDRANT_VOLUME: &str = "know-qdrant-data";

/// Push the qdrant data to Docker Hub as an image
pub async fn push(name: &str) -> Result<()> {
    // Validate image name format
    if !name.contains('/') {
        anyhow::bail!(
            "Image name must include repository (e.g., 'myuser/my-knowledge:v1')"
        );
    }

    println!("Preparing to push knowledge base to {}...", name);

    // Step 1: Create a snapshot of the qdrant volume
    let temp_dir = std::env::temp_dir().join("know-push");
    std::fs::create_dir_all(&temp_dir)?;

    println!("Creating snapshot of qdrant data...");

    // Use docker to copy volume data to a tar file
    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "-v",
            &format!("{}:/data:ro", QDRANT_VOLUME),
            "-v",
            &format!("{}:/backup", temp_dir.display()),
            "alpine",
            "tar",
            "-czf",
            "/backup/snapshot.tar.gz",
            "-C",
            "/data",
            ".",
        ])
        .output()
        .await
        .context("Failed to create volume snapshot")?;

    if !output.status.success() {
        anyhow::bail!(
            "Failed to create snapshot: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Step 2: Create a Dockerfile for the snapshot
    let dockerfile_path = temp_dir.join("Dockerfile");
    std::fs::write(
        &dockerfile_path,
        r#"FROM scratch
COPY snapshot.tar.gz /snapshot.tar.gz
LABEL org.opencontainers.image.title="know knowledge base"
LABEL org.opencontainers.image.description="Qdrant vector database snapshot created by know"
"#,
    )?;

    // Step 3: Build the image
    println!("Building image...");
    let status = Command::new("docker")
        .args(["build", "-t", name, "."])
        .current_dir(&temp_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("Failed to build image")?;

    if !status.success() {
        anyhow::bail!("Failed to build image");
    }

    // Step 4: Push to Docker Hub
    println!("Pushing to Docker Hub...");
    let status = Command::new("docker")
        .args(["push", name])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("Failed to push image")?;

    if !status.success() {
        anyhow::bail!("Failed to push image. Make sure you're logged in with 'docker login'");
    }

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).ok();

    println!("\nSuccessfully pushed knowledge base to {}", name);
    println!("Others can now use: know pull {}", name);

    Ok(())
}

/// Pull a knowledge base from Docker Hub
pub async fn pull(name: &str) -> Result<()> {
    println!("Pulling knowledge base from {}...", name);

    // Step 1: Pull the image
    let status = Command::new("docker")
        .args(["pull", name])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("Failed to pull image")?;

    if !status.success() {
        anyhow::bail!("Failed to pull image");
    }

    // Step 2: Extract the snapshot from the image
    let temp_dir = std::env::temp_dir().join("know-pull");
    std::fs::create_dir_all(&temp_dir)?;

    println!("Extracting snapshot...");

    // Create a container and copy the snapshot out
    let output = Command::new("docker")
        .args(["create", "--name", "know-temp-extract", name])
        .output()
        .await
        .context("Failed to create temporary container")?;

    if !output.status.success() {
        anyhow::bail!(
            "Failed to create container: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Copy snapshot from container
    let snapshot_path = temp_dir.join("snapshot.tar.gz");
    let status = Command::new("docker")
        .args([
            "cp",
            "know-temp-extract:/snapshot.tar.gz",
            &snapshot_path.to_string_lossy(),
        ])
        .status()
        .await
        .context("Failed to copy snapshot from container")?;

    // Remove temporary container
    Command::new("docker")
        .args(["rm", "know-temp-extract"])
        .output()
        .await
        .ok();

    if !status.success() {
        anyhow::bail!("Failed to extract snapshot from image");
    }

    // Step 3: Restore to qdrant volume
    println!("Restoring to qdrant volume...");

    // Ensure volume exists
    Command::new("docker")
        .args(["volume", "create", QDRANT_VOLUME])
        .output()
        .await
        .ok();

    // Extract snapshot to volume
    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "-v",
            &format!("{}:/data", QDRANT_VOLUME),
            "-v",
            &format!("{}:/backup:ro", temp_dir.display()),
            "alpine",
            "sh",
            "-c",
            "rm -rf /data/* && tar -xzf /backup/snapshot.tar.gz -C /data",
        ])
        .output()
        .await
        .context("Failed to restore snapshot")?;

    if !output.status.success() {
        anyhow::bail!(
            "Failed to restore snapshot: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).ok();

    println!("\nSuccessfully pulled knowledge base from {}", name);
    println!("Run 'know up' to start using it.");

    Ok(())
}
