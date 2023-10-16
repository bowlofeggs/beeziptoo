//! beeziptoo
//!
//! Because we wanted to implement `bzip2`, too.
use thiserror::Error as ThisError;

mod rle;

/// These are the possible errors that can occur during decompression.
#[derive(Debug, ThisError)]
pub enum DecompressError {
    /// The runlength decoder encountered an invalid input.
    #[error("Failed to decode at a runlength step")]
    RunLengthDecode,
}

impl From<rle::Error> for DecompressError {
    fn from(value: rle::Error) -> Self {
        match value {
            rle::Error::RunLengthTruncated => DecompressError::RunLengthDecode,
        }
    }
}

/// Compress the given data.
pub fn compress(data: &[u8]) -> Vec<u8> {
    rle::forward(data)
}

/// Decompress the given data.
///
/// # Errors
///
/// This function is failable since it is possible the given data isn't a valid `bzip2` archive.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, DecompressError> {
    rle::reverse(data).map_err(|e| e.into())
}
