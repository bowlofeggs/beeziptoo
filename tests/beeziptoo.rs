//! Test the beeziptoo public interface.
use std::io::Read;

use beeziptoo::{compress, decompress};

/// Assert that a roundtrip of compression and decompression results in the same data.
///
/// This is marked should_panic because it won't pass until we can ser/de the file format.
///
/// https://github.com/bowlofeggs/beeziptoo/issues/10
#[test]
#[should_panic]
fn roundtrip() {
    let data = b"ABBCCCDDDDEEEEEFFFFFFGGGGGGGHHHHHHHH";

    let mut compressed_data = compress(&data[..]).expect("Could not compress data");
    let mut result = decompress(&mut compressed_data).expect("Could not decompress data");

    let mut all_data = vec![];
    result
        .read_to_end(&mut all_data)
        .expect("Could not read decompressed data");
    assert_eq!(all_data, data);
}
