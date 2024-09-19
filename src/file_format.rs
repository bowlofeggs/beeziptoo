// Copyright ® 2023-2024 Andrew Halle and Randy Barlow
//! Read and write the bzip2 file format.

// TODO remove
#![allow(unused)]

use std::io::Read;

use self::bitstream::Bitstream;
use crate::huffman::HuffmanCodedData;

mod bitstream;

/// Errors that can occur when decoding bzip2 streams.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Invalid block size
    #[error("Invalid block size")]
    InvalidBlockSize,

    /// Invalid footer magic
    #[error("The footer magic should be the BCD-encoded square root of pi, but was not.")]
    InvalidFooterMagic,

    /// Invalid header
    #[error("The header should be the bytes BZh, but was not.")]
    InvalidHeader,

    /// IOError while reading the input.
    #[error("IOError: {0}")]
    IOError(#[from] std::io::Error),

    /// Unexpected end of stream.
    #[error("Unexpected end of stream.")]
    UnexpectedEof,

    /// Invalid block header (BCD pi)
    #[error("The block header should be BCD-coded pi.")]
    InvalidBlockHeader,

    /// Invalid `randomized` bit.
    ///
    /// The `Randomized` field is a single bit, and in bzip2 should always be 0. This error means
    /// it was 1, which is unexpected.
    #[error("The Randomized field should be 0, but was 1.")]
    InvalidRandomizedField,
}

// = Parser ====================================================================

struct Parser<R> {
    bitstream: Bitstream<R>,
}

impl<R> Parser<R>
where
    R: Read,
{
    fn new<B>(bitstream: B) -> Self
    where
        B: Into<Bitstream<R>>,
    {
        Self {
            bitstream: bitstream.into(),
        }
    }

    fn parse(mut self) -> Result<BZipFile, DecodeError> {
        let stream = self.stream()?;

        debug_assert!(
            self.bitstream.is_empty()?,
            "The parser failed to read the entire stream."
        );

        Ok(BZipFile { stream })
    }

    fn stream(&mut self) -> Result<BZipStream, DecodeError> {
        let header = self.stream_header()?;
        let footer = self.stream_footer()?;

        Ok(BZipStream { header, footer })
    }

    /// Read the stream footer
    fn stream_footer(&mut self) -> Result<StreamFooter, DecodeError> {
        Ok(StreamFooter {
            magic: self.footer_magic()?,
            crc: self.stream_crc()?,
            padding: self.footer_padding()?,
        })
    }

    fn stream_header(&mut self) -> Result<StreamHeader, DecodeError> {
        Ok(StreamHeader {
            magic: self.header_magic()?,
            version: self.version()?,
            level: self.level()?,
        })
    }

    /// Get the stream CRC from the footer.
    fn stream_crc(&mut self) -> Result<StreamCrc, DecodeError> {
        let crc: u32 = self.bitstream.get_integer(32)?;

        Ok(StreamCrc(crc))
    }

    fn footer_magic(&mut self) -> Result<FooterMagic, DecodeError> {
        let footer: u64 = self.bitstream.get_integer(48)?;

        if footer != 0x177245385090 {
            return Err(DecodeError::InvalidFooterMagic);
        }

        Ok(FooterMagic)
    }

    fn footer_padding(&mut self) -> Result<Padding, DecodeError> {
        let padding: Vec<bitstream::Bit> = self.bitstream.get_padding();

        Ok(Padding(padding))
    }

    fn header_magic(&mut self) -> Result<HeaderMagic, DecodeError> {
        let header: u16 = self.bitstream.get_integer(16)?;
        if header != 0x425a {
            return Err(DecodeError::InvalidHeader);
        }

        Ok(HeaderMagic)
    }

    fn version(&mut self) -> Result<Version, DecodeError> {
        let version: u8 = self.bitstream.get_integer(8)?;
        if version != b'h' {
            return Err(DecodeError::InvalidHeader);
        }

        Ok(Version)
    }

    fn level(&mut self) -> Result<Level, DecodeError> {
        let level: u8 = self.bitstream.get_integer(8)?;
        if level < b'1' || level > b'9' {
            return Err(DecodeError::InvalidBlockSize);
        }

        Ok(Level(level - b'1' + 1))
    }
}

// =============================================================================

// = File format structs =======================================================

#[derive(Debug)]
struct BZipFile {
    stream: BZipStream,
}

#[derive(Debug)]
struct BZipStream {
    header: StreamHeader,
    // Bring these back when we are ready for them, which the universe will reveal in time
    // (spacetime).
    //block: Vec<StreamBlock>,
    footer: StreamFooter,
}

#[derive(Debug)]
struct StreamHeader {
    magic: HeaderMagic,
    version: Version,
    level: Level,
}

#[derive(Debug)]
struct StreamBlock {
    header: BlockHeader,
    trees: BlockTrees,
    data: BlockData,
}

#[derive(Debug)]
struct BlockHeader {
    magic: BlockMagic,
    crc: BlockCrc,
    randomized: Randomized,
    orig_ptr: OriginPointer,
}

#[derive(Debug)]
struct BlockTrees {
    sym_map: SymbolMap,
    trees: Vec<Tree>,
    selectors: Vec<Selector>,
}

#[derive(Debug)]
struct StreamFooter {
    magic: FooterMagic,
    crc: StreamCrc,
    padding: Padding,
}

#[derive(Debug)]
struct Level(u8);

#[derive(Debug)]
struct HeaderMagic;
#[derive(Debug)]
struct Version;
#[derive(Debug)]
struct BlockMagic;
#[derive(Debug)]
struct BlockCrc;
#[derive(Debug)]
struct Randomized;
#[derive(Debug)]
struct OriginPointer;
#[derive(Debug)]
struct SymbolMap;
#[derive(Debug)]
struct Tree;
#[derive(Debug)]
struct Selector;
#[derive(Debug)]
struct BlockData;
#[derive(Debug)]
struct FooterMagic;
#[derive(Debug)]
struct StreamCrc(u32);
#[derive(Debug)]
struct Padding(Vec<bitstream::Bit>);

// =============================================================================

pub fn decode(bytes: &[u8]) -> Result<HuffmanCodedData, DecodeError> {
    let mut stream = bitstream::Bitstream::new(bytes);
    let mut parser = Parser::new(stream);

    let bzip_file = parser.parse()?;

    dbg!(bzip_file);

    // Maybe this could be a bzip_file.into()?
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

fn bcd_pi(bytes: &[u8]) -> Result<&[u8], DecodeError> {
    if bytes.len() < 6 {
        return Err(DecodeError::UnexpectedEof);
    }

    match bytes {
        [0x31, 0x41, 0x59, 0x26, 0x53, 0x59, rest @ ..] => Ok(rest),
        _ => Err(DecodeError::InvalidBlockHeader),
    }
}

// TODO: CRC32 needs to be validated somewhere.
//
// This might be a free function, or it might be a method on a hypothetical `Block` type.

fn crc32(bytes: &[u8]) -> Result<(u32, &[u8]), DecodeError> {
    if bytes.len() < 4 {
        return Err(DecodeError::UnexpectedEof);
    }

    let (crc, rest) = bytes.split_at(4);
    // TODO: Figure out if little-endian is the correct endianness.
    let crc = u32::from_le_bytes(crc.try_into().unwrap());
    Ok((crc, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the Parser
    mod parser {
        use super::*;

        /// Test the `parse()` method.
        mod parse {
            use super::*;

            /// Test that we can parse the empty file.
            mod empty {
                use super::*;

                /// Test with a level 1 compressed file.
                #[test]
                fn level_1() {
                    let input = b"\x42\x5a\x68\x31\x17\x72\x45\x38\x50\x90\x00\x00\x00\x00";
                    let parser = Parser::new(&input[..]);

                    let file = parser.parse().expect("This should not fail to parse");

                    assert_eq!(file.stream.header.level.0, 1);
                    panic!("What kind of assertions can we make?");
                }

                /// Test with a level 9 compressed file.
                #[test]
                fn level_9() {
                    let input = b"\x42\x5a\x68\x39\x17\x72\x45\x38\x50\x90\x00\x00\x00\x00";
                    let parser = Parser::new(&input[..]);

                    let file = parser.parse().expect("This should not fail to parse");

                    assert_eq!(file.stream.header.level.0, 9);
                    panic!("What kind of assertions can we make?");
                }
            }
        }
    }

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

            if let Err(DecodeError::InvalidHeader) = decoded {
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
