use std::{
    fmt,
    io::{self, Read},
};

/// A bit.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Bit {
    Zero,
    One,
}

const BUFFER_SIZE: usize = 512;

/// An adapter over a reader that turns a byte slice into an iterator of bits.
pub(crate) struct Bitstream<R> {
    /// The reader.
    inner: R,
    /// The bytes we've read which still need handling.
    buffer: Box<[u8; BUFFER_SIZE]>,
    /// The number of bytes that we have in the buffer.
    buffer_size: usize,
    /// The index of the byte we're about to handle.
    buffer_pointer: usize,
    /// The index of the bit that we are about to handle.
    // This pointer views bit 0 as the rightmost bit and bit 7 as the leftmost bit.
    bit_pointer: u8,
}

impl<R> fmt::Debug for Bitstream<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bitstream")
            .field("buffer_size", &self.buffer_size)
            .field("buffer_pointer", &self.buffer_pointer)
            .field("bit_pointer", &self.bit_pointer)
            .finish()
    }
}

pub trait FromBits {
    fn from_bits(value: &[Bit]) -> Self;
}

macro_rules! impl_from_bits {
    ($kind:ty, $size:expr) => {
        impl FromBits for $kind {
            fn from_bits(value: &[Bit]) -> Self {
                debug_assert!(
                    value.len() <= $size,
                    concat!("Cannot convert {} bits into a ", stringify!($kind)),
                    value.len()
                );
                debug_assert!(
                    value.len() > 0,
                    "Cannot build a {} out of 0 bits",
                    stringify!($kind)
                );

                let mut x = 0;
                for bit in value {
                    x <<= 1;

                    match bit {
                        Bit::Zero => {
                            // Nothing to do
                        }
                        Bit::One => {
                            x |= 1;
                        }
                    }
                }

                x
            }
        }
    };
}

impl_from_bits!(u8, 8);
impl_from_bits!(u16, 16);
impl_from_bits!(u32, 32);
impl_from_bits!(u64, 64);

/// TODO
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
struct Peek {
    /// TODO
    bits: Vec<Bit>,
    /// TODO
    new_buffer_pointer: usize,
    /// TODO
    new_bit_pointer: u8,
}

impl<R> Bitstream<R>
where
    R: Read,
{
    pub(crate) fn new(inner: R) -> Bitstream<R> {
        Bitstream {
            inner,
            buffer: Box::new([0; 512]),
            buffer_size: 0,
            buffer_pointer: 0,
            bit_pointer: 7,
        }
    }

    pub(super) fn get_integer<T>(&mut self, num_bits_to_read: u8) -> io::Result<T>
    where
        T: FromBits,
    {
        let bits: Vec<Bit> = (0..num_bits_to_read)
            .map(|_| self.get_next_bit())
            .collect::<Result<Vec<_>, io::Error>>()?;

        Ok(T::from_bits(&bits))
    }

    /// Read the next integer of the requested number of bits without moving the cursor.
    ///
    /// It is an error to set `num_bits_to_read` to `0`. In debug mode this will panic. In release
    /// mode, this will return a `0`.
    pub(super) fn peek_integer<T>(&mut self, num_bits_to_read: u8) -> io::Result<T>
    where
        T: FromBits,
    {
        let Peek { bits, .. } = self.peek_n_bits(num_bits_to_read.into())?;

        Ok(T::from_bits(&bits))
    }

    /// Read n bits without moving the bit pointer.
    ///
    /// This will return `UnexpectedEof` if there are fewer than `n` bits left to read.
    ///
    /// # Panics
    ///
    /// - If n > 8 * (BUFFER_SIZE - 1).
    fn peek_n_bits(&mut self, n: usize) -> io::Result<Peek> {
        assert!(
            // The -1 is necessary because we may have an incompletely consumed byte that we need
            // to keep.
            n <= 8 * (BUFFER_SIZE - 1),
            "n must be less than or equal to {} but was {}",
            8 * (BUFFER_SIZE - 1),
            n
        );

        if self.bits_in_buffer() < n {
            self.shift_and_read()?;
        }
        if self.bits_in_buffer() < n {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "couldn't read enough bits",
            ));
        }

        let mut bits = vec![];
        let mut new_bit_pointer = self.bit_pointer;
        let mut new_buffer_pointer = self.buffer_pointer;

        for _ in 0..n {
            let bit = if (1 << new_bit_pointer) & self.buffer[new_buffer_pointer] == 0 {
                Bit::Zero
            } else {
                Bit::One
            };
            bits.push(bit);

            if new_bit_pointer == 0 {
                new_buffer_pointer += 1;
                new_bit_pointer = 7;
            } else {
                new_bit_pointer -= 1;
            }
        }

        Ok(Peek {
            bits,
            new_buffer_pointer,
            new_bit_pointer,
        })
    }

    /// Returns how many bits are left in the buffer.
    fn bits_in_buffer(&self) -> usize {
        if self.buffer_size == self.buffer_pointer {
            return 0;
        }

        let next_byte_to_read = self.buffer_pointer + 1;
        let totally_unread_bits = (self.buffer_size - next_byte_to_read) * 8;

        (self.bit_pointer as usize + 1) + totally_unread_bits
    }

    pub(crate) fn get_next_bit(&mut self) -> io::Result<Bit> {
        let peek = self.peek_n_bits(1)?;

        self.bit_pointer = peek.new_bit_pointer;
        self.buffer_pointer = peek.new_buffer_pointer;

        Ok(peek.bits[0])
    }

    /// Shift the bytes in the buffer to the front of the buffer and fill the rest by attempting to
    /// read from the underlying reader.
    fn shift_and_read(&mut self) -> io::Result<()> {
        self.buffer
            .copy_within(self.buffer_pointer..self.buffer_size, 0);
        self.buffer_size -= self.buffer_pointer;
        self.buffer_pointer = 0;
        self.buffer_size += self.inner.read(&mut self.buffer[self.buffer_size..])?;

        Ok(())
    }

    /// Sync up the bitstream with a byte boundary.
    ///
    /// This function reads and returns enough bits out of the bitstream so that there are no
    /// partially consumed bytes in the buffer.
    pub(super) fn get_padding(&mut self) -> Vec<Bit> {
        let bits_to_read = if self.buffer_pointer == self.buffer_size {
            0
        } else {
            self.bit_pointer + 1
        };

        if bits_to_read == 0 {
            return vec![];
        }

        (0..bits_to_read)
            .map(|_| self.get_next_bit())
            .collect::<Result<Vec<_>, _>>()
            .expect("All bits should have been Some but they were not")
    }

    /// Return `true` if there are no bits that haven't been consumed.
    pub(super) fn is_empty(&mut self) -> io::Result<bool> {
        let peek = self.peek_n_bits(1);
        match peek {
            Ok(_) => Ok(false),
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => Ok(true),
            Err(err) => Err(err),
        }
    }
}

impl<R> From<R> for Bitstream<R>
where
    R: Read,
{
    fn from(value: R) -> Self {
        Bitstream::new(value)
    }
}

impl<R> Iterator for Bitstream<R>
where
    R: Read,
{
    type Item = io::Result<Bit>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.get_next_bit() {
            Ok(bit) => Some(Ok(bit)),
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => None,
            Err(err) => Some(Err(err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let input: &[u8] = &[10];

        let bitstream = Bitstream::new(input);

        let output: io::Result<Vec<Bit>> = bitstream.collect();

        assert_eq!(
            output.unwrap(),
            &[
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::Zero,
                Bit::One,
                Bit::Zero,
                Bit::One,
                Bit::Zero
            ]
        );
    }

    #[test]
    fn get_integer() {
        let input: [u8; 4] = 0x0074740e_u32.to_be_bytes();
        let mut bitstream = Bitstream::new(&input[..]);

        let value: u32 = bitstream.get_integer(24).expect("This should fit in a u32");

        assert_eq!(value, 0x7474);
        assert_eq!(bitstream.bit_pointer, 7);
        assert_eq!(bitstream.buffer_pointer, 3);
    }

    #[test]
    fn get_padding() {
        let input: &[u8] = &[10];

        let mut bitstream = Bitstream::new(input);
        let _: u8 = bitstream.get_integer(4).unwrap();

        assert_eq!(
            bitstream.get_padding(),
            &[Bit::One, Bit::Zero, Bit::One, Bit::Zero]
        );
        assert_eq!(bitstream.bit_pointer, 7);
        assert_eq!(bitstream.buffer_pointer, 1);
    }

    /// Test the shift_and_read() method.
    mod shift_and_read {
        use super::*;

        /// Test when we are on the very last byte already.
        #[test]
        fn last_byte() {
            let input: Vec<u8> = (0..BUFFER_SIZE * 2).map(|v| (v % 256) as u8).collect();
            let mut bitstream = Bitstream::new(input.as_slice());
            // Let's set the state such that we have read all but the last byte in the buffer.
            (0..BUFFER_SIZE - 1)
                .map(|_| bitstream.get_integer::<u8>(8))
                .collect::<Vec<_>>();
            // Let's also leave the bit pointer in a more exicing position than 0.
            let _: u8 = bitstream.get_integer(3).unwrap();

            // Now let's ask it to shift and read, which should move that integer to the front and
            // should refill the buffer.
            bitstream.shift_and_read().unwrap();

            assert_eq!(bitstream.buffer[..4], [255, 0, 1, 2]);
            assert_eq!(bitstream.bit_pointer, 4);
            assert_eq!(bitstream.buffer_pointer, 0);
        }

        /// Test when we are on the third-to-last byte.
        #[test]
        fn third_to_last_byte() {
            let input: Vec<u8> = (0..BUFFER_SIZE * 2).map(|v| (v % 256) as u8).collect();
            let mut bitstream = Bitstream::new(input.as_slice());
            // Let's set the state such that we have read all but the last byte in the buffer.
            (0..BUFFER_SIZE - 3)
                .map(|_| bitstream.get_integer::<u8>(8))
                .collect::<Vec<_>>();
            // Let's also leave the bit pointer in a more exicing position than 0.
            let _: u8 = bitstream.get_integer(4).unwrap();

            // Now let's ask it to shift and read, which should move that integer to the front and
            // should refill the buffer.
            bitstream.shift_and_read().unwrap();

            assert_eq!(bitstream.buffer[..4], [253, 254, 255, 0]);
            assert_eq!(bitstream.bit_pointer, 3);
            assert_eq!(bitstream.buffer_pointer, 0);
        }
    }

    mod bits_in_buffer {
        use super::*;

        #[test]
        fn empty() {
            let bs = Bitstream::new(&b""[..]);
            assert_eq!(bs.bits_in_buffer(), 0);
        }

        #[test]
        fn two_read() {
            let mut bs = Bitstream::new(&b"\x01\x10"[..]);
            let _ = bs.get_integer::<u16>(2);
            assert_eq!(bs.bits_in_buffer(), 14);
        }

        #[test]
        fn one_left() {
            let mut bs = Bitstream::new(&b"\x01\x10"[..]);
            let _ = bs.get_integer::<u16>(15);
            assert_eq!(bs.bits_in_buffer(), 1);
        }

        #[test]
        fn fully_read() {
            let mut bs = Bitstream::new(&b"\x01\x10"[..]);
            let _ = bs.get_integer::<u16>(16);
            assert_eq!(bs.bits_in_buffer(), 0);
        }
    }

    /// Test [`Bitstream::peek_integer`].
    mod peek_integer {
        use super::*;

        #[test]
        fn empty() {
            let mut bs = Bitstream::new(&b""[..]);

            let result: io::Result<u8> = bs.peek_integer(1);

            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        }

        #[test]
        #[should_panic = "Cannot build a u8 out of 0 bits"]
        #[cfg(debug_assertions)]
        fn zero() {
            let mut bs = Bitstream::new(&b""[..]);

            let res: u8 = bs.peek_integer(0).unwrap();
        }

        #[test]
        #[cfg(not(debug_assertions))]
        fn zero() {
            let mut bs = Bitstream::new(&b""[..]);

            let res: u8 = bs.peek_integer(0).unwrap();

            // In release mode this returns 0 instead of panicking
            assert_eq!(res, 0);
        }

        #[test]
        fn one() {
            let mut bs = Bitstream::new(&b"\x80"[..]);

            let result: u8 = bs.peek_integer(1).unwrap();

            assert_eq!(result, 1);
        }

        #[test]
        fn two() {
            let mut bs = Bitstream::new(&b"\x80"[..]);

            let result: u8 = bs.peek_integer(2).unwrap();

            assert_eq!(result, 2);
        }

        #[test]
        fn cross_boundary() {
            let mut bs = Bitstream::new(&b"\x80\x80"[..]);
            let _ = bs.get_integer::<u8>(7);

            let result: u8 = bs.peek_integer(2).unwrap();

            assert_eq!(result, 1);
        }

        #[test]
        fn last() {
            let mut bs = Bitstream::new(&b"\x00\x01"[..]);
            let _ = bs.get_integer::<u16>(15);

            let result: u8 = bs.peek_integer(1).unwrap();

            assert_eq!(result, 1);
        }

        #[test]
        fn error_beyond_end() {
            let mut bs = Bitstream::new(&b"\x00"[..]);
            let _ = bs.get_integer::<u8>(8);

            let result: io::Result<u8> = bs.peek_integer(1);

            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        }
    }

    mod peek_n_bits {
        use super::*;

        #[test]
        fn empty() {
            let mut bs = Bitstream::new(&b""[..]);
            assert_eq!(
                bs.peek_n_bits(1).unwrap_err().kind(),
                io::ErrorKind::UnexpectedEof
            );
        }

        #[test]
        fn zero() {
            let mut bs = Bitstream::new(&b""[..]);
            assert_eq!(
                bs.peek_n_bits(0).unwrap(),
                Peek {
                    bits: vec![],
                    new_buffer_pointer: 0,
                    new_bit_pointer: 7
                }
            );
        }

        #[test]
        fn one() {
            let mut bs = Bitstream::new(&b"\x80"[..]);
            assert_eq!(
                bs.peek_n_bits(1).unwrap(),
                Peek {
                    bits: vec![Bit::One],
                    new_buffer_pointer: 0,
                    new_bit_pointer: 6,
                }
            );
        }

        #[test]
        fn two() {
            let mut bs = Bitstream::new(&b"\x80"[..]);
            assert_eq!(
                bs.peek_n_bits(2).unwrap(),
                Peek {
                    bits: vec![Bit::One, Bit::Zero],
                    new_buffer_pointer: 0,
                    new_bit_pointer: 5,
                }
            );
        }

        #[test]
        fn cross_boundary() {
            let mut bs = Bitstream::new(&b"\x80\x80"[..]);
            let _ = bs.get_integer::<u8>(7);
            assert_eq!(
                bs.peek_n_bits(2).unwrap(),
                Peek {
                    bits: vec![Bit::Zero, Bit::One],
                    new_buffer_pointer: 1,
                    new_bit_pointer: 6,
                }
            );
        }

        #[test]
        fn last() {
            let mut bs = Bitstream::new(&b"\x00\x01"[..]);
            let _ = bs.get_integer::<u16>(15);
            assert_eq!(
                bs.peek_n_bits(1).unwrap(),
                Peek {
                    bits: vec![Bit::One],
                    new_buffer_pointer: 2,
                    new_bit_pointer: 7,
                }
            );
        }

        #[test]
        fn error_beyond_end() {
            let mut bs = Bitstream::new(&b"\x00"[..]);
            let _ = bs.get_integer::<u8>(8);
            assert_eq!(
                bs.peek_n_bits(1).unwrap_err().kind(),
                io::ErrorKind::UnexpectedEof
            );
        }
    }
}
