#![no_main]

use std::io::Read;

use libfuzzer_sys::fuzz_target;

use beeziptoo::{compress, decompress};

fuzz_target!(|data: &[u8]| {
    let mut compressed = compress(&data[..]).expect("Could not read input data");

    let mut decompressed_reader = decompress(&mut compressed)
        .expect("We should never fail to decompress something we compressed");
    let mut decompressed = Vec::new();
    decompressed_reader.read_to_end(&mut decompressed).expect("Could not read decompressed data");
    assert_eq!(decompressed, data);
});
