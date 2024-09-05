use std::io::{self, Read};

/// A bit.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum Bit {
    Zero,
    One,
}

/// An adapter over a reader that turns a byte slice into an iterator of bits.
pub(super) struct Bitstream<R> {
    /// The reader.
    inner: R,
    /// The bytes we've read which still need handling.
    buffer: Box<[u8; 512]>,
    /// The number of bytes that we have in the buffer.
    buffer_size: u16,
    /// The index of the byte we're about to handle.
    buffer_pointer: u16,
    /// The index of the bit that we are about to handle.
    // This pointer views bit 0 as the rightmost bit and bit 7 as the leftmost bit.
    bit_pointer: u8,
}

trait FromBits {
    fn from_bits(value: &[Bit]) -> Self;
}

macro_rules! impl_from_bits {
    ($kind:ty, $size:expr) => {
        impl FromBits for $kind {
            fn from_bits(value: &[Bit]) -> Self {
                assert!(
                    value.len() <= $size,
                    concat!("Cannot convert {} bits into a ", stringify!($kind)),
                    value.len()
                );

                let mut x = 0;
                for bit in value {
                    x = x << 1;

                    match bit {
                        Bit::Zero => {
                            // Nothing to do
                        }
                        Bit::One => {
                            x = x | 1;
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

impl<R> Bitstream<R>
where
    R: Read,
{
    pub(super) fn new(inner: R) -> Bitstream<R> {
        Bitstream {
            inner,
            buffer: Box::new([0; 512]),
            buffer_size: 0,
            buffer_pointer: 0,
            bit_pointer: 7,
        }
    }

    pub(super) fn get_integer<T>(&mut self, num_bits_to_read: u8) -> Result<T, super::DecodeError>
    where
        T: FromBits,
    {
        let bits: Vec<Bit> = (0..num_bits_to_read)
            .map(|_| {
                self.get_next_bit()
                    .map_err(|e| e.into())
                    .and_then(|b| b.ok_or_else(|| super::DecodeError::UnexpectedEof))
            })
            .collect::<Result<Vec<_>, super::DecodeError>>()?;

        Ok(T::from_bits(&bits))
    }

    pub(super) fn get_next_bit(&mut self) -> io::Result<Option<Bit>> {
        if self.buffer_pointer == self.buffer_size {
            debug_assert_eq!(self.bit_pointer, 7);
            self.buffer_size = self.inner.read(&mut self.buffer[..])?.try_into().unwrap();
            self.buffer_pointer = 0;
        }

        if self.buffer_size == 0 {
            return Ok(None);
        }

        let bit = if (1 << self.bit_pointer) & self.buffer[self.buffer_pointer as usize] == 0 {
            Bit::Zero
        } else {
            Bit::One
        };

        if self.bit_pointer == 0 {
            self.buffer_pointer += 1;
            self.bit_pointer = 7;
        } else {
            self.bit_pointer -= 1;
        }

        Ok(Some(bit))
    }
}

impl<R> Iterator for Bitstream<R>
where
    R: Read,
{
    type Item = io::Result<Bit>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_bit().transpose()
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
    }
}
