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
        Commands::Up => docker::up().await,
        Commands::Down => docker::down().await,
        Commands::Ingest { path, extensions } => {
            ingest::ingest(&cli, path, extensions).await
        }
        Commands::Ask { query } => {
            let question = query.join(" ");
            ingest::ask(&cli, &question).await
        }
        Commands::Serve { port } => server::serve(&cli, *port).await,
        Commands::Clean { collection } => qdrant::clean(collection).await,
        Commands::Push { name } => registry::push(name).await,
        Commands::Pull { name } => registry::pull(name).await,
        Commands::Status => docker::status().await,
    }
}
