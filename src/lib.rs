//! beeziptoo
//!
//! Because we wanted to implement `bzip2`, too.
use std::io::{self, Cursor, Read};

mod burrows_wheeler;
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
    /// The runlength decoder encountered an invalid input.
    #[error("Failed to decode at a runlength step")]
    RunLengthDecode,
    /// The burrows-wheeler decoder encountered an invalid input.
    #[error("Failed to decode at a burrows-wheeler step")]
    BurrowsWheelerDecode,
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

impl From<burrows_wheeler::DecodeError> for DecompressError {
    fn from(_value: burrows_wheeler::DecodeError) -> Self {
        DecompressError::BurrowsWheelerDecode
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
    let burrows_wheeler_data = burrows_wheeler::encode(&rle_data);
    let move_to_front_data = move_to_front::encode(&burrows_wheeler_data);
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
    let mut _all_data = vec![];
    data.read_to_end(&mut _all_data)?;

    // This is a stub to satisfy that huffman decoding needs a valid HuffmanCodeData. We need to
    // put something here that can read a bzip file and produce this type.
    let huffman_data = huffman::HuffmanCodedData::default();
    let un_huffman_data = huffman::decode(&huffman_data)?;
    let un_rle2 = rle2::decode(&un_huffman_data);
    let un_move_to_front_data = move_to_front::decode(&un_rle2);
    let un_burrows_wheeler_data = burrows_wheeler::decode(&un_move_to_front_data)?;
    let un_rle_data = rle1::decode(&un_burrows_wheeler_data)?;

    let cursor = Cursor::new(un_rle_data);

    Ok(cursor)
}
