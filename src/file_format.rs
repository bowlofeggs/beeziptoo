// Copyright ® 2023-2024 Andrew Halle and Randy Barlow
//! Read and write the bzip2 file format.

use crate::huffman::HuffmanCodedData;

/// Errors that can occur when decoding bzip2 streams.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Invalid block size
    #[error("Invalid block size")]
    InvalidBlockSize,

    /// Invalid header
    #[error("The header should be the bytes BZh, but was not.")]
    InvalidHeader,

    /// Unexpected end of stream.
    #[error("Unexpected end of stream.")]
    UnexpectedEof,
}

pub fn decode(bytes: &[u8]) -> Result<HuffmanCodedData, DecodeError> {
    let bytes = validate_header(bytes)?;
    let (block_size, bytes) = block_size(bytes)?;

    Ok(HuffmanCodedData::default())
}

/// The block size of the uncompressed data, in bytes.
#[derive(Debug, PartialEq)]
struct BlockSize(u32);

impl BlockSize {
    fn new(block_size: u8) -> Result<Self, DecodeError> {
        if block_size < b'1' || block_size > b'9' {
            return Err(DecodeError::InvalidBlockSize);
        }

        let expanded_block_size = ((block_size - b'1' + 1) as u32) * 100_000;

        Ok(Self(expanded_block_size))
    }
}

fn block_size(bytes: &[u8]) -> Result<(BlockSize, &[u8]), DecodeError> {
    if bytes.is_empty() {
        return Err(DecodeError::UnexpectedEof);
    }

    Ok((BlockSize::new(bytes[0])?, &bytes[1..]))
}

fn validate_header(bytes: &[u8]) -> Result<&[u8], DecodeError> {
    if bytes.len() < 3 {
        return Err(DecodeError::UnexpectedEof);
    }

    match bytes {
        [b'B', b'Z', b'h', rest @ ..] => Ok(rest),
        _ => Err(DecodeError::InvalidHeader),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test block size reading.
    mod read_block_size {
        use super::*;

        /// A valid block size.
        #[test]
        fn valid() {
            let input = b"3test";

            let (block_size, bytes) = block_size(input).expect("The block size should be valid");

            assert_eq!(block_size, BlockSize(300_000));
            assert_eq!(bytes, b"test");
        }
    }

    /// Test header validation.
    mod header_validation {
        use super::*;

        /// Just right. A valid header, followed by other data.
        #[test]
        fn just_right() {
            let input = b"BZhtest";

            let decoded = validate_header(input).expect("This should pass validation.");

            assert_eq!(decoded, b"test");
        }

        /// Too short.
        #[test]
        fn too_short() {
            let input = b"12";

            let decoded = decode(input);

            if let Err(DecodeError::UnexpectedEof) = decoded {
                // This is the expected outcome
            } else {
                panic!("This should have returned an error.");
            }
        }

        /// Wrong header.
        #[test]
        fn wrong_header() {
            // The first three bytes should be BZh. BZ for bzip, and h for Huffman.
            let input = b"BZi";

            let decoded = decode(input);

            if let Err(DecodeError::InvalidHeader) = decoded {
                // This is the expected outcome
            } else {
                panic!("This should have returned an error.");
            }
        }
    }
}
