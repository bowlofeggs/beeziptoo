//! beeziptoo
//!
//! Because we wanted to implement `bzip2`, too.
use std::io::{self, Cursor, Read};

mod burrows_wheeler;
mod rle;

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
}

impl From<io::Error> for DecompressError {
    fn from(value: io::Error) -> Self {
        DecompressError::IOError(value)
    }
}

impl From<rle::Error> for DecompressError {
    fn from(_value: rle::Error) -> Self {
        DecompressError::RunLengthDecode
    }
}

impl From<burrows_wheeler::DecodeError> for DecompressError {
    fn from(_value: burrows_wheeler::DecodeError) -> Self {
        DecompressError::BurrowsWheelerDecode
    }
}

/// Compress the given data.
pub fn compress<R>(mut data: R) -> Result<impl Read, CompressError>
where
    R: Read,
{
    let mut all_data = vec![];
    data.read_to_end(&mut all_data)?;

    let rle_data = rle::encode(&all_data);
    let burrows_wheeler_data = burrows_wheeler::encode(&rle_data);

    let cursor = Cursor::new(burrows_wheeler_data);

    Ok(cursor)
}

/// Decompress the given data.
///
/// # Errors
///
/// This function is failable since it is possible the given data isn't a valid `bzip2` archive.
pub fn decompress<R>(data: &mut R) -> Result<impl Read, DecompressError>
where
    R: Read,
{
    let mut all_data = vec![];
    data.read_to_end(&mut all_data)?;

    let un_burrows_wheeler_data = burrows_wheeler::decode(&all_data)?;
    let un_rle_data = rle::decode(&un_burrows_wheeler_data)?;

    let cursor = Cursor::new(un_rle_data);

    Ok(cursor)
}
