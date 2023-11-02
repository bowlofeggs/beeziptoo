#![no_main]

use std::io::Read;

use libfuzzer_sys::fuzz_target;

use beeziptoo::{compress, decompress};

fuzz_target!(|data: &[u8]| {
    if let Ok(mut decompressed_reader) = decompress(&data[..]) {
        let mut compressed_reader = compress(&mut decompressed_reader)
            .expect("We should never fail to compress");
        let mut compressed = Vec::new();
        compressed_reader.read_to_end(&mut compressed).expect("Could not read compressed data");

        // It is not always true that the compressed data will match the original input. Consider
        // [10, 10, 10, 10, 93, 10] vs [10, 10, 10, 10, 94] - they are equivalent, and the first is
        // valid input but re-compressing would result in the second. So we'll just make a weak
        // assertion that the re-compressed data has a length when the original data did.
        if data.len() > 0 {
            assert!(compressed.len() > 0);
        }
    }
});
