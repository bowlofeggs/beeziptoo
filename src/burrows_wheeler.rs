use std::collections::VecDeque;

pub(super) fn encode(data: &[u8]) -> Vec<u8> {
    todo!()
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
