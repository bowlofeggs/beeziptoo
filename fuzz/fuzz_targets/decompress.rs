#![no_main]

use std::io::Read;
use std::io::Write;
use std::process::Command;
use std::process::Stdio;

use libfuzzer_sys::fuzz_target;

use beeziptoo::{compress, decompress};

// Ideas:
// * have the fuzzer randomly choose some cli flags for real bzip2
fuzz_target!(|data: &[u8]| {
    let mut child = Command::new("bzip2")
        .arg("-c")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    std::thread::scope(|s| {
        s.spawn(move || {
            stdin.write_all(data).unwrap();
        });
    });
    let bytes = child.wait_with_output().unwrap().stdout;

    let mut decompressed = decompress(&bytes[..]).unwrap();
    let mut buf = vec![];
    let decompressed = decompressed.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, data);
});
