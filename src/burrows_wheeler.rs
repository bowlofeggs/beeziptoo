use std::collections::VecDeque;

/// Encode with the Burrows-Wheeler Transform.
///
/// # Notes
///
/// The origin pointer (index of original row in the sorted rotation block) is appended to the end of the
/// data as 3 little-endian bytes.
pub(super) fn encode(data: &[u8]) -> Vec<u8> {
    // ASSUMPTION max block size for this stage is 900 kB
    assert!(data.len() < 900_000);

    if data.is_empty() {
        return Vec::new();
    }

    let mut all_rotations = all_rotations(data);
    all_rotations.sort_unstable();
    let origin_pointer = all_rotations
        .binary_search_by_key(&data, |v| &v[..])
        .expect("`data` must be in `all_rotations`");
    let mut output: Vec<u8> = all_rotations
        .into_iter()
        .map(|mut v| {
            v.pop()
                .expect("presence of inner `Vec` means data was not empty")
        })
        .collect();
    // ASSUMPTION we're encoding the origin pointer as a little-endian index.
    output.extend_from_slice(&origin_pointer.to_le_bytes()[..3]);

    output
}

pub(super) fn decode(data: &[u8]) -> Vec<u8> {
    todo!()
}

fn all_rotations(data: &[u8]) -> Vec<Vec<u8>> {
    let mut deque: VecDeque<u8> = data.into_iter().copied().collect();
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

    mod encode {
        use super::*;

        #[test]
        fn small() {
            let input = b"cdab";

            let encoded = encode(input);

            assert_eq!(&encoded[..4], b"dabc");
            assert_eq!(&encoded[4..], &[2, 0, 0]);
        }

        #[test]
        fn empty() {
            let encoded = encode(&[]);

            assert_eq!(encoded, &[]);
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
