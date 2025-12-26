# beeziptoo

We can do bzip, too.

This is a pure Rust implementation of the bzip2 compression algorithm.

## Usage

Here's an example:

```rust
use std::{io::{Read, Write}, process::{Command, Stdio}};

use beeziptoo::decompress;

let peter_piper = "If Peter Piper picked a peck of pickled peppers, where's the peck of pickled peppers Peter Piper picked?????";

let mut child = Command::new("bzip2")
    .arg("-c")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .spawn()
    .unwrap();
{
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(peter_piper.as_bytes()).unwrap();
}
let bytes = child.wait_with_output().unwrap().stdout;

let mut data = decompress(&bytes[..]).expect("Cannot decompress test data");

let mut buffer = vec![];
let _num_bytes = data
    .read_to_end(&mut buffer)
    .expect("Cannot read decompressed data");
assert_eq!(std::str::from_utf8(&buffer).unwrap(), peter_piper);
```

## Project status

`beeziptoo` currently supports decompression, and should be able to handle any
`bzip2` file. There are still a few undone TODOs, such as CRC checking.

It does not yet support compression (see
<https://github.com/bowlofeggs/beeziptoo/issues/23>). Much of the compression
code has been written, but file serialization is still a TODO. Since we cannot
yet serialize a file, it is also difficult to assert whether the existing
compression code is correct or not, as an ideal test would be to see if the
canonical `bzip2` program can decompress a file that we write.
