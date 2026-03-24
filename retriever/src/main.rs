mod schema;
mod tokenizers;
mod indexer;
mod search;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Build {
        #[arg(long)] docs_jsonl: PathBuf,
        #[arg(long)] index_dir: PathBuf,
    },
    Serve {
        #[arg(long)] index_dir: PathBuf,
        #[arg(long, default_value_t = 4107)] port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();
    match cli.command {
        Command::Build { docs_jsonl, index_dir } => indexer::build_index(&docs_jsonl, &index_dir),
        Command::Serve { index_dir, port } => search::serve(index_dir, port).await,
    }
}
