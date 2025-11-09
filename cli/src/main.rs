// Copyright ® 2023-2024 Andrew Halle and Randy Barlow
//! beeziptoo
//!
//! Because we wanted to implement `bzip2`, too.
//!
//! This is the CLI.

use std::{fs::OpenOptions, path::PathBuf};

use anyhow::Context;
use clap::Parser;

use beeziptoo::decompress;

#[derive(Parser)]
struct Cli {
    /// The path to be unzipped.
    path: PathBuf,

    /// The path to write to.
    destination: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let bytes = std::fs::File::open(&cli.path)
        .with_context(|| format!("Unable to open path {}", cli.path.display()))?;

    let mut unpacked_bytes = decompress(&bytes)
        .with_context(|| format!("Unable to decompress path {}", cli.path.display()))?;

    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&cli.destination)
        .with_context(|| format!("Unable to write to path {}", cli.destination.display()))?;
    std::io::copy(&mut unpacked_bytes, &mut writer)
        .with_context(|| format!("Unable to write to path {}", cli.destination.display()))?;

    Ok(())
}
