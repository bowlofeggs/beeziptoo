// Copyright ® 2023-2025 Andrew Halle and Randy Barlow
//! Read and write the bzip2 file format.

// TODO remove
#![allow(unused)]

use std::io::{self, ErrorKind, Read};

use bitstream::Bit;

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

    /// Invalid block header (BCD pi)
    #[error("The block header should be BCD-coded pi.")]
    InvalidBlockHeader,

    /// Invalid `randomized` bit.
    ///
    /// The `Randomized` field is a single bit, and in bzip2 should always be 0. This error means
    /// it was 1, which is unexpected.
    #[error("The Randomized field should be 0, but was 1.")]
    InvalidRandomizedField,

    /// Invalid `Selector`.
    #[error("The selector should be a zero-terminated string of length at most 6")]
    InvalidSelector,

    /// Invalid `Tree`.
    #[error("The tree data structure could not be deserialized")]
    InvalidTree,
}

impl DecodeError {
    fn unexpected_eof(msg: &'static str) -> Self {
        Self::IOError(io::Error::new(ErrorKind::UnexpectedEof, msg))
    }
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

    // TODO: Should this return an iterator over blocks instead of a `Vec<StreamBlock>`.
    //
    // This is one of the things that precludes us from streaming the file.
    fn blocks(&mut self) -> Result<Vec<StreamBlock>, DecodeError> {
        let mut blocks = vec![];

        while let Some(block) = self.next_block()? {
            blocks.push(block);
        }

        Ok(blocks)
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
        let blocks = self.blocks()?;
        let footer = self.stream_footer()?;

        Ok(BZipStream {
            header,
            blocks,
            footer,
        })
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
        if !(b'1'..=b'9').contains(&level) {
            return Err(DecodeError::InvalidBlockSize);
        }

        Ok(Level(level - b'1' + 1))
    }

    fn next_block(&mut self) -> Result<Option<StreamBlock>, DecodeError> {
        let maybe_magic: u64 = self.bitstream.peek_integer(48)?;

        if maybe_magic != 0x314159265359 {
            return Ok(None);
        }

        let header = self.block_header()?;
        let trees = self.block_trees()?;
        let data = self.block_data()?;

        Ok(Some(StreamBlock {
            header,
            trees,
            data,
        }))
    }

    fn block_header(&mut self) -> Result<BlockHeader, DecodeError> {
        let magic = self.bitstream.get_integer(48)?;
        let crc = self.bitstream.get_integer(32)?;
        let randomized = self.bitstream.get_integer(1)?;
        let origin_pointer = self.bitstream.get_integer(24)?;

        if randomized != 0 {
            return Err(DecodeError::InvalidRandomizedField);
        }

        Ok(BlockHeader {
            magic: BlockMagic(magic),
            crc: BlockCrc(crc),
            randomized: Randomized(randomized),
            orig_ptr: OriginPointer(origin_pointer),
        })
    }

    fn symbol_map(&mut self) -> Result<SymbolMap, DecodeError> {
        let l1: u16 = self.bitstream.get_integer(16)?;
        let mut l2 = vec![];

        {
            let mut l1 = l1.reverse_bits();
            while l1 != 0 {
                if l1 % 2 != 0 {
                    l2.push(self.bitstream.get_integer(16)?);
                }
                l1 >>= 1;
            }
        }

        Ok(SymbolMap { l1, l2 })
    }

    /// Parse a single tree.
    // TODO: Test this
    fn tree(&mut self, num_symbols: u16) -> Result<Tree, DecodeError> {
        let mut tree = vec![];
        let mut initial_bit_length: u8 = self.bitstream.get_integer(5)?;

        for _ in 0..num_symbols {
            while self.bitstream.peek_integer::<u8>(1)? == 1 {
                let delta: u8 = self.bitstream.get_integer(2)?;
                match delta {
                    2 => initial_bit_length += 1,
                    3 => initial_bit_length -= 1,
                    _ => unreachable!(),
                }
            }

            let terminator: u8 = self.bitstream.get_integer(1)?;
            if terminator != 0 {
                return Err(DecodeError::InvalidTree);
            }

            tree.push(initial_bit_length);
        }

        // The code that goes with these lengths is defined in https://www.ietf.org/rfc/rfc1951.txt
        Ok(Tree(tree))
    }

    fn selector(&mut self) -> Result<Selector, DecodeError> {
        const MAX_SELECTOR_BITS: u8 = 6;

        let mut bits = vec![];

        for i in 0..MAX_SELECTOR_BITS {
            let bit = self.bitstream.get_next_bit()?;

            if bit == Bit::Zero {
                break;
            }

            if i == MAX_SELECTOR_BITS - 1 {
                return Err(DecodeError::InvalidSelector);
            }

            bits.push(bit);
        }

        Ok(Selector(bits.len() as u8))
    }

    fn block_trees(&mut self) -> Result<BlockTrees, DecodeError> {
        let sym_map = self.symbol_map()?;
        let num_symbols = sym_map.num_symbols();

        let num_trees: u8 = self.bitstream.get_integer(3)?;
        let num_selectors: u16 = self.bitstream.get_integer(15)?;

        let mut selectors = vec![];
        for _ in 0..num_selectors {
            selectors.push(self.selector()?);
        }

        let mut trees = vec![];
        for _ in 0..num_trees {
            trees.push(self.tree(num_symbols)?);
        }

        Ok(BlockTrees {
            sym_map,
            trees,
            selectors,
        })
    }

    fn block_data(&mut self) -> Result<BlockData, DecodeError> {
        // TODONEXT: Figure out how long the block data is
        todo!()
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
    blocks: Vec<StreamBlock>,
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
struct BlockMagic(u64);

#[derive(Debug)]
struct BlockCrc(u32);

#[derive(Debug)]
struct Randomized(u8);

#[derive(Debug)]
struct OriginPointer(u32);

#[derive(Debug)]
struct SymbolMap {
    l1: u16,
    l2: Vec<u16>,
}

impl SymbolMap {
    // TODO test this
    fn num_symbols(&self) -> u16 {
        // The spec says that num_syms is num_stack + 2
        self.l2.iter().map(u16::count_ones).sum() + 2
    }
}

#[derive(Debug)]
struct Selector(u8);

#[derive(Debug)]
struct Tree(Vec<u8>);

#[derive(Debug)]
struct HeaderMagic;
#[derive(Debug)]
struct Version;
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
        if !(b'1'..=b'9').contains(&block_size) {
            return Err(DecodeError::InvalidBlockSize);
        }

        let expanded_block_size = ((block_size - b'1' + 1) as u32) * 100_000;

        Ok(Self(expanded_block_size))
    }
}

fn block_size(bytes: &[u8]) -> Result<(BlockSize, &[u8]), DecodeError> {
    if bytes.is_empty() {
        return Err(DecodeError::unexpected_eof("there were no bytes"));
    }

    Ok((BlockSize::new(bytes[0])?, &bytes[1..]))
}

fn validate_header(bytes: &[u8]) -> Result<&[u8], DecodeError> {
    if bytes.len() < 3 {
        return Err(DecodeError::unexpected_eof("there were fewer than 3 bytes"));
    }

    match bytes {
        [b'B', b'Z', b'h', rest @ ..] => Ok(rest),
        _ => Err(DecodeError::InvalidHeader),
    }
}

fn bcd_pi(bytes: &[u8]) -> Result<&[u8], DecodeError> {
    if bytes.len() < 6 {
        return Err(DecodeError::unexpected_eof("there were fewer than 6 bytes"));
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
        return Err(DecodeError::unexpected_eof("there were fewer than 4 bytes"));
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
                    assert_eq!(file.stream.blocks.len(), 0);
                    assert_eq!(file.stream.footer.crc.0, 0);
                }

                /// Test with a level 9 compressed file.
                #[test]
                fn level_9() {
                    let input = b"\x42\x5a\x68\x39\x17\x72\x45\x38\x50\x90\x00\x00\x00\x00";
                    let parser = Parser::new(&input[..]);

                    let file = parser.parse().expect("This should not fail to parse");

                    assert_eq!(file.stream.header.level.0, 9);
                    assert_eq!(file.stream.blocks.len(), 0);
                    assert_eq!(file.stream.footer.crc.0, 0);
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
