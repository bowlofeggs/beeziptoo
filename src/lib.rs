// Copyright ® 2023-2024 Andrew Halle and Randy Barlow
//! beeziptoo
//!
//! Because we wanted to implement `bzip2`, too.
use std::io::{self, Cursor, Read};

use crate::burrows_wheeler::BwtEncoded;

mod burrows_wheeler;
mod file_format;
mod huffman;
mod move_to_front;
mod rle1;
mod rle2;

/// These are the possible errors that can occur during compression.
#[derive(Debug, thiserror::Error)]
pub enum CompressError {
    /// An IO error occurred.
    #[error("I/O error: {0}")]
    IOError(io::Error),
}

impl From<io::Error> for CompressError {
    fn from(value: io::Error) -> Self {
        CompressError::IOError(value)
    }
}

/// These are the possible errors that can occur during decompression.
#[derive(Debug, thiserror::Error)]
pub enum DecompressError {
    /// An IO error occurred.
    #[error("I/O error: {0}")]
    IOError(io::Error),
    /// Unable to parse the bzip2 stream.
    #[error("Unable to parse the bzip2 stream: {0}")]
    Parse(#[from] file_format::DecodeError),
    /// The runlength decoder encountered an invalid input.
    #[error("Failed to decode at a runlength step")]
    RunLengthDecode,
    /// The burrows-wheeler decoder encountered an invalid input.
    #[error("Failed to decode at a burrows-wheeler step")]
    BurrowsWheelerDecode(#[from] burrows_wheeler::DecodeError),
    /// The huffman decoder encountered an invalid input.
    #[error("Failed to decode at a huffman code step")]
    HuffmanDecode,
}

impl From<io::Error> for DecompressError {
    fn from(value: io::Error) -> Self {
        DecompressError::IOError(value)
    }
}

impl From<rle1::Error> for DecompressError {
    fn from(value: rle1::Error) -> Self {
        match value {
            rle1::Error::RunLengthInvalid(_) => DecompressError::RunLengthDecode,
            rle1::Error::RunLengthTruncated => DecompressError::RunLengthDecode,
        }
    }
}

impl From<huffman::Error> for DecompressError {
    fn from(_value: huffman::Error) -> Self {
        Self::HuffmanDecode
    }
}

/// Compress the given data.
pub fn compress<R>(mut data: R) -> Result<impl Read, CompressError>
where
    R: Read,
{
    let mut all_data = vec![];
    data.read_to_end(&mut all_data)?;

    let rle_data = rle1::encode(&all_data);
    // TODO: Origin pointer is unused here. When we write out the file in the correct format, we
    // will use it then.
    let burrows_wheeler_data = burrows_wheeler::encode(&rle_data);
    let move_to_front_data = move_to_front::encode(&burrows_wheeler_data.data);
    let rle2_data = rle2::encode(&move_to_front_data);
    let _huffman_data = huffman::encode(&rle2_data);
    // This is a stub to satisfy the return type. We need to put something here that can turn
    // huffman_data into the bzip2 file format.
    let file_data = Vec::new();

    let cursor = Cursor::new(file_data);

    Ok(cursor)
}

/// Decompress the given data.
///
/// # Errors
///
/// This function is failable since it is possible the given data isn't a valid `bzip2` archive.
pub fn decompress<R>(mut data: R) -> Result<impl Read, DecompressError>
where
    R: Read,
{
    let mut all_data = vec![];
    let mut decompressed_data = vec![];
    data.read_to_end(&mut all_data)?;

    let un_file_data = file_format::decode(&all_data)?;

    for block in &un_file_data {
        let un_huffman_data = huffman::decode(block.symbols());
        let un_rle2 = rle2::decode(&un_huffman_data);
        let un_move_to_front_data = move_to_front::decode(&un_rle2, block.symbol_stack());
        let un_burrows_wheeler_data = burrows_wheeler::decode(&BwtEncoded::new(
            un_move_to_front_data,
            block.origin_pointer(),
        ))?;
        let mut un_rle_data = rle1::decode(&un_burrows_wheeler_data)?;
        decompressed_data.append(&mut un_rle_data);
    }

    let cursor = Cursor::new(decompressed_data);

    Ok(cursor)
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write as _,
        process::{Command, Stdio},
    };

    use super::*;

    #[test]
    fn can_we_read() {
        let peter_piper = "If Peter Piper picked a peck of pickled peppers, where's the peck of pickled peppers Peter Piper picked?????";

        let mut child = Command::new("bzip2")
            .args(&["-c"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        {
            let mut stdin = child.stdin.take().unwrap();
            stdin.write_all(peter_piper.as_bytes()).unwrap();
        }
        let bytes = child.wait_with_output().unwrap().stdout;

        let mut data = decompress(&bytes[..]).expect("Cannot decompress test data");

        let mut buffer = vec![];
        let _bytes = data
            .read_to_end(&mut buffer)
            .expect("Cannot read decompressed data");
        assert_eq!(std::str::from_utf8(&buffer).unwrap(), peter_piper);
    }
}
