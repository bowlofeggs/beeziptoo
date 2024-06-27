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

    fn get_next_bit(&mut self) -> io::Result<Option<Bit>> {
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
}
