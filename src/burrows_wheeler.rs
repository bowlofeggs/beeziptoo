//! Define the Burrows-Wheeler encode and decode steps.
use std::collections::VecDeque;

use crate::file_format::OriginPointer;

/// Stores BWT-encoded information.
#[derive(Debug, Default, Clone)]
pub(crate) struct BwtEncoded {
    /// The encoded data.
    pub(crate) data: Vec<u8>,

    /// The index of the original row in the sorted rotation block.
    ///
    /// This is required for decoding.
    pub(crate) origin_pointer: OriginPointer,
}

impl BwtEncoded {
    pub(crate) fn new(data: Vec<u8>, origin_pointer: OriginPointer) -> Self {
        BwtEncoded {
            data,
            origin_pointer,
        }
    }
}

/// Encode with the Burrows-Wheeler Transform.
///
/// # Notes
///
/// The origin pointer (index of original row in the sorted rotation block) is appended to the end of the
/// data as 3 little-endian bytes.
pub(super) fn encode(data: &[u8]) -> BwtEncoded {
    // ASSUMPTION max block size for this stage is 900 kB
    assert!(data.len() < 900_000);

    if data.is_empty() {
        return BwtEncoded::default();
    }

    let mut all_rotations = all_rotations(data);
    all_rotations.sort_unstable();
    let origin_pointer: OriginPointer = all_rotations
        .binary_search_by_key(&data, |v| &v[..])
        .expect("`data` must be in `all_rotations`")
        .try_into()
        .expect("`origin_pointer` must fit into 24 bits");
    let data: Vec<u8> = all_rotations
        .into_iter()
        .map(|mut v| {
            v.pop()
                .expect("presence of outer `Vec` means inner `Vec` was not empty")
        })
        .collect();

    BwtEncoded {
        data,
        origin_pointer,
    }
}

/// Errors that can occur when decoding a Burrows-Wheeler array.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("origin pointer out of range")]
    InvalidOriginPointer,
}

/// Decode the Burrows-Wheeler transform.
///
/// # Notes
///
/// If the input to this function is wrong, it is possible for this function to produce a row that
/// is not a rotation of the input. The algorithm is quite sensitive to the encoder and decoder
/// using the same sort ordering, and if they do not it is possible to get a row that is missing
/// characters.
pub(super) fn decode(
    BwtEncoded {
        data,
        origin_pointer,
    }: &BwtEncoded,
) -> Result<Vec<u8>, DecodeError> {
    // ASSUMPTION it is valid for this to be empty
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let origin_pointer: usize = origin_pointer.try_into().unwrap();
    if origin_pointer > data.len() - 1 {
        return Err(DecodeError::InvalidOriginPointer);
    }

    // TODO benchmark `Vec` instead of `VecDeque`.
    let mut table: Vec<VecDeque<u8>> = vec![VecDeque::new(); data.len()];
    for _ in 0..data.len() {
        for (byte, row) in data.iter().zip(table.iter_mut()) {
            row.push_front(*byte);
        }
        table.sort_unstable();
    }

    let output = table.swap_remove(origin_pointer);
    Ok(output.into())
}

/// Generate every possible rotation of the given data.
fn all_rotations(data: &[u8]) -> Vec<Vec<u8>> {
    let mut deque: VecDeque<u8> = data.iter().copied().collect();
    let mut output = Vec::with_capacity(deque.len());
    for _ in 0..deque.len() {
        output.push(deque.iter().copied().collect());
        deque.rotate_left(1);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let input = b"adlfjasldjfaslkfdsjaklsd";

        let result = decode(&encode(input)).unwrap();

        assert_eq!(input.as_slice(), result);
    }

    mod decode {
        use super::*;

        /// This tests the example from
        /// <https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform>.
        #[test]
        fn banana() {
            // Wikipedia uses a $, which sorts last in their example. However, in ASCII the $
            // character will sort first. As a result, we can't use the example as is and have it
            // work. It just so happens that we can use lower case s as a substitute, because it
            // will sort first out of this set of characters.
            let input = BwtEncoded {
                data: b"BNN^AAsA".to_vec(),
                origin_pointer: 6.into(),
            };

            let decoded = decode(&input).unwrap();

            // Note above that we substituted an s for $ so we sort the same as Wikipedia.
            assert_eq!(decoded, b"^BANANAs");
        }

        #[test]
        fn small() {
            let input = BwtEncoded {
                data: b"dabc".to_vec(),
                origin_pointer: 2.into(),
            };

            let decoded = decode(&input).unwrap();

            assert_eq!(decoded, b"cdab");
        }

        /// Test with empty data.
        #[test]
        fn empty() {
            let decoded = decode(&BwtEncoded::default()).unwrap();

            assert_eq!(decoded, &[]);
        }
    }
    mod encode {
        use super::*;

        #[test]
        fn small() {
            let input = b"cdab";

            let encoded = encode(input);

            assert_eq!(encoded.data, b"dabc");
            let origin_pointer: usize = encoded.origin_pointer.try_into().unwrap();
            assert_eq!(origin_pointer, 2);
        }

        #[test]
        fn empty() {
            let encoded = encode(&[]);

            assert_eq!(encoded.data, BwtEncoded::default().data);
        }
    }

    mod all_rotations {
        use super::*;

        #[test]
        fn small() {
            let input = b"abcd";

            let rotations = all_rotations(input);

            assert!(rotations.iter().any(|v| v == b"abcd"));
            assert!(rotations.iter().any(|v| v == b"bcda"));
            assert!(rotations.iter().any(|v| v == b"cdab"));
            assert!(rotations.iter().any(|v| v == b"dabc"));
        }

        #[test]
        fn empty() {
            let rotations = all_rotations(&[]);

            assert!(rotations.is_empty());
        }
    }
}
