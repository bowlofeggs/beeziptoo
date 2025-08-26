//! Implement the [move-to-front] algorithm.
//!
//! [move-to-front]: https://en.wikipedia.org/wiki/Move-to-front_transform

use crate::file_format::SymbolStack;

/// Convert the data into the move-to-front encoded format.
pub(super) fn encode(data: &[u8]) -> Vec<u8> {
    // This list contains all the possible bytes that can be found in data. As we encounter each
    // byte in input, we find it in this list and the index in the list is encoded into the output.
    // Then the value is move to the front of the list. Over time, this results in the frequently
    // used bytes tending towards the front of the list, and the infrequently used bytes tending
    // towards the back.
    let mut symbols: Vec<u8> = (0..=255).collect();
    let mut output = vec![];

    for byte in data {
        let (index, _) = symbols
            .iter()
            .enumerate()
            .find(|(_, value)| *value == byte)
            .expect("Every possible byte should be here");
        // This as should be safe because we know there are only 256 values in symbols, so the
        // index should always encodable as a u8.
        output.push(index as u8);
        let value = symbols.remove(index);
        symbols.insert(0, value);
    }

    output
}

/// Convert the move-to-front encoded data back to the original data.
///
/// The `symbol_stack` contains all the possible values that can be found in the output data. As we
/// encounter each byte in input, we use it as an index into this list in this list and the value
/// at that index is encoded into the output. Then the value is move to the front of the list.
pub(super) fn decode(data: &[u8], SymbolStack(mut symbols): SymbolStack) -> Vec<u8> {
    let mut output = vec![];

    for index in data {
        // This should be safe because on all platforms a u8 should always be safe to convert to a
        // usize.
        let index = *index as usize;
        let value = symbols[index];
        output.push(value);
        let value = symbols.remove(index);
        symbols.insert(0, value);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_decode {
        use super::*;

        #[test]
        fn banana() {
            let symbol_stack = SymbolStack((0..=255).collect());
            let input = [1, 1, 13, 1, 1, 1];

            let decoded = decode(&input, symbol_stack);

            // This is basically "banana" if you map a-z to 0-25.
            assert_eq!(decoded, [1, 0, 13, 0, 13, 0]);
        }

        /// The fuzzer found a crash when 255 was in the input.
        #[test]
        fn input_255() {
            let symbol_stack = SymbolStack((0..=255).collect());
            let input = [255];

            let decoded = decode(&input, symbol_stack);

            assert_eq!(decoded, [255]);
        }
    }

    mod test_encode {
        use super::*;

        #[test]
        fn banana() {
            // This is basically "banana" if you map a-z to 0-25.
            let input = [1, 0, 13, 0, 13, 0];

            let encoded = encode(&input);

            assert_eq!(encoded, [1, 1, 13, 1, 1, 1]);
        }

        /// The fuzzer found a crash when 255 was in the input.
        #[test]
        fn input_255() {
            let input = [255];

            let encoded = encode(&input);

            assert_eq!(encoded, [255]);
        }
    }
}
