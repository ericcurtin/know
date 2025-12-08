// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 The Know Authors

mod backend;
mod cli;
mod docker;
mod ingest;
mod qdrant;
mod registry;
mod server;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Run { query } => {
            docker::ensure_running().await?;
            let question = query.join(" ");
            ingest::run(&cli, &question).await
        }
        Commands::Ingest { path, extensions } => {
            docker::ensure_running().await?;
            ingest::ingest(&cli, path, extensions).await
        }
        Commands::Serve { port } => {
            docker::ensure_running().await?;
            server::serve(&cli, *port).await
        }
        Commands::Down => docker::down().await,
        Commands::Clean { collection } => qdrant::clean(collection).await,
        Commands::Push { name } => registry::push(name).await,
        Commands::Pull { name } => registry::pull(name).await,
        Commands::Status => docker::status().await,
    }
}
