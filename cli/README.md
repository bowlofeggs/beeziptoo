# beeziptoo-cli

We can do bzip, too.

This is a very simple `bzip2` CLI, written in pure Rust.

## Usage

Here's an example:

```
$ beeziptoo-cli example.bz2 example
```

## Project status

`beeziptoo-cli` currently supports decompression, and should be able to handle
any `bzip2` file. There are still a few undone TODOs, such as CRC checking.

It does not yet support compression (see
<https://github.com/bowlofeggs/beeziptoo/issues/23>).

It is not a drop-in replacement for the canonical `bzip2` utilities, though we
may attempt that goal in the future.
