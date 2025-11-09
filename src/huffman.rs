//! Define the huffman encoding functions.

use std::collections::{BinaryHeap, HashMap};

use super::rle2;
use crate::file_format::bitstream::Bitstream;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid node index")]
    InvalidNodeIndex,

    #[error("IoError: {0}")]
    IoError(#[from] std::io::Error),

    #[error("The bitstream was truncated")]
    TruncatedBitstream,
}

#[derive(Debug, Default)]
#[expect(unused)] // TODO: Should be used when we encode.
pub(super) struct HuffmanCodedData {
    /// This is the set of trees used in the encoded data.
    trees: Vec<tree::Tree>,
    /// This contains the encoded data.
    blocks: Vec<HuffmanBlock>,
}

#[derive(Debug, Default)]
#[expect(unused)] // TODO: Should be used when we encode.
struct HuffmanBlock {
    /// Points into [`HuffmanCodedData::trees`] to indicate which tree was used for this block.
    tree_index: u8,
    /// This is the huffman-coded data.
    bitvec: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub(crate) enum Symbol {
    RunA,
    RunB,
    Byte(u16),
    Eob,
}

impl From<&rle2::Symbol> for Symbol {
    fn from(value: &rle2::Symbol) -> Self {
        match value {
            rle2::Symbol::RunA => Self::RunA,
            rle2::Symbol::RunB => Self::RunB,
            // TODO: We used to add 1 on this line, in 20d0518aa8c5bb26b7489aabdb8ad763d2f5dfda, but we
            // found that there was a bug where we were off by 1 in the symbol values. There was
            // some dicussion around whether this transformation should do that, or perhaps whether
            // something else should be responsible for doing it. Perhaps consider whether we do
            // want this +1 here for cleanliness.
            rle2::Symbol::Byte(byte) => Self::Byte(*byte as u16),
        }
    }
}

impl From<&Symbol> for rle2::Symbol {
    fn from(value: &Symbol) -> Self {
        match value {
            Symbol::RunA => rle2::Symbol::RunA,
            Symbol::RunB => rle2::Symbol::RunB,
            // TODO: We used to subtract 1 here. See TODO in the above From impl going the other
            // direction.
            Symbol::Byte(byte) => rle2::Symbol::Byte((*byte) as u8),
            // We'll strip off the EOB symbol before calling this function.
            Symbol::Eob => unreachable!(),
        }
    }
}

pub(crate) mod tree {
    use crate::file_format::bitstream::Bit;

    use super::*;

    use std::{cmp::Reverse, io::Read};

    type SymbolBitMap = HashMap<Symbol, Vec<bool>>;

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct NodeIdx(usize);

    #[derive(Eq, PartialEq)]
    struct SymbolWeight(usize, NodeIdx);

    impl Ord for SymbolWeight {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0)
        }
    }

    impl PartialOrd for SymbolWeight {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    #[derive(Debug)]
    enum Node {
        Branch(NodeIdx, NodeIdx),
        Leaf(Symbol),
    }

    #[derive(Debug)]
    pub(crate) struct Tree {
        root: NodeIdx,
        nodes: Vec<Node>,
        symbol_bitmap: SymbolBitMap,
    }

    impl Tree {
        pub(crate) fn new(symbol_weights: HashMap<Symbol, usize>) -> Self {
            let mut nodes: Vec<Node> = Vec::new();
            let mut priority_queue: BinaryHeap<_> = symbol_weights
                .iter()
                .map(|(symbol, weight)| {
                    let idx = nodes.len();
                    nodes.push(Node::Leaf(*symbol));
                    Reverse(SymbolWeight(*weight, NodeIdx(idx)))
                })
                .collect();

            while priority_queue.len() > 1 {
                let Reverse(SymbolWeight(first_weight, right)) = priority_queue
                    .pop()
                    .expect("We asserted that there were two");
                let Reverse(SymbolWeight(second_weight, left)) = priority_queue
                    .pop()
                    .expect("We asserted that there were two");
                let idx = nodes.len();
                nodes.push(Node::Branch(left, right));
                priority_queue.push(Reverse(SymbolWeight(
                    first_weight + second_weight,
                    NodeIdx(idx),
                )));
            }

            // our priority queue has exactly one item
            // that means our huffman tree is in our nodes list at index priority_queue.pop()
            let Reverse(SymbolWeight(_, root)) = priority_queue
                .pop()
                .expect("there must be exactly one item in the queue");

            let symbol_bitmap = Self::build_symbol_bitmap(root, &nodes);
            Self {
                root,
                nodes,
                symbol_bitmap,
            }
        }

        pub(crate) fn encode(&self, symbols: &[rle2::Symbol]) -> Vec<u8> {
            let mut bitvec = Vec::new();

            for symbol in symbols {
                let path = self
                    .symbol_bitmap
                    .get(&symbol.into())
                    .expect("All symbols should be in the tree");
                bitvec.extend(path.iter().map(|b| if *b { 1 } else { 0 }));
            }

            bitvec
        }

        /// Decode one `Symbol` from the `bitstream`.
        pub(crate) fn decode<R>(&self, bitstream: &mut Bitstream<R>) -> Result<Symbol, Error>
        where
            R: Read,
        {
            let mut location = self.root;

            loop {
                let node = self.nodes.get(location.0).ok_or(Error::InvalidNodeIndex)?;
                match node {
                    Node::Leaf(symbol) => {
                        return Ok(*symbol);
                    }
                    Node::Branch(left, right) => {
                        let next_bit = bitstream.get_next_bit()?;

                        match next_bit {
                            Bit::Zero => location = *left,
                            Bit::One => location = *right,
                        }
                    }
                }
            }
        }

        fn build_symbol_bitmap(root: NodeIdx, nodes: &[Node]) -> SymbolBitMap {
            let mut symbol_bitmap = HashMap::new();

            // Depth-first search
            let mut worklist: Vec<(NodeIdx, Vec<bool>)> = vec![(root, Vec::new())];
            while let Some((NodeIdx(idx), path)) = worklist.pop() {
                let node = &nodes[idx];
                match *node {
                    Node::Branch(left, right) => {
                        let mut right_path = path.clone();
                        right_path.push(true);
                        worklist.push((right, right_path));
                        let mut left_path = path;
                        left_path.push(false);
                        worklist.push((left, left_path));
                    }
                    Node::Leaf(symbol) => {
                        symbol_bitmap.insert(symbol, path);
                    }
                }
            }

            symbol_bitmap
        }
    }

    #[derive(Debug)]
    pub(crate) struct InvalidBitmap;

    impl TryFrom<Vec<u8>> for Tree {
        type Error = InvalidBitmap;

        fn try_from(value: Vec<u8>) -> Result<Self, Self::Error> {
            let symbol_bitmap = canonical_huffman_table(&value);
            symbol_bitmap.try_into()
        }
    }

    impl TryFrom<SymbolBitMap> for Tree {
        type Error = InvalidBitmap;
        fn try_from(symbol_bitmap: SymbolBitMap) -> Result<Self, Self::Error> {
            #[derive(Debug, Copy, Clone)]
            enum TempNode {
                Branch(Option<NodeIdx>, Option<NodeIdx>),
                Leaf(Symbol),
            }

            impl Default for TempNode {
                fn default() -> Self {
                    TempNode::Branch(None, None)
                }
            }

            impl TryFrom<TempNode> for Node {
                type Error = InvalidBitmap;

                fn try_from(value: TempNode) -> Result<Self, Self::Error> {
                    let node = match value {
                        TempNode::Branch(Some(left), Some(right)) => Node::Branch(left, right),
                        TempNode::Leaf(symbol) => Node::Leaf(symbol),
                        _ => return Err(InvalidBitmap),
                    };

                    Ok(node)
                }
            }

            let mut nodes = vec![TempNode::default()];
            let root = NodeIdx(0);

            for (symbol, path) in &symbol_bitmap {
                if path.is_empty() {
                    return Err(InvalidBitmap);
                }

                nodes.push(TempNode::Leaf(*symbol));
                let node_idx = NodeIdx(nodes.len() - 1);

                let mut curr = root;
                // We chop off the last bit here because all the nodes we deal with in this loop
                // ought to be branches. The last bit should end us up on a node that is a leaf
                // node (the node we created above) so we handle that separately.
                for bit in &path[..path.len() - 1] {
                    // We want to work on a copy of the node so that we don't borrow all nodes
                    // mutably. We need to do this because we might need to push a new node in the
                    // case that we encounter a branch node that hasn't been created yet.
                    let old_idx = curr.0;
                    let mut node = nodes[old_idx];
                    match &mut node {
                        TempNode::Leaf(_) => return Err(InvalidBitmap),
                        TempNode::Branch(_, Some(right)) if *bit => {
                            curr = *right;
                        }
                        TempNode::Branch(Some(left), _) if !bit => {
                            curr = *left;
                        }
                        TempNode::Branch(left, right) => {
                            nodes.push(TempNode::default());
                            let node_idx = NodeIdx(nodes.len() - 1);
                            curr = node_idx;
                            if *bit {
                                *right = Some(node_idx);
                            } else {
                                *left = Some(node_idx);
                            }
                        }
                    }
                    // Here we put the copy of the node that we mutated back in `nodes`.
                    nodes[old_idx] = node;
                }

                let node = &mut nodes[curr.0];
                // We returned an error early if `path` was empty.
                match (node, path.last().unwrap()) {
                    (TempNode::Leaf(_), _) => return Err(InvalidBitmap),
                    (TempNode::Branch(left, right), bit) => {
                        if *bit {
                            *right = Some(node_idx);
                        } else {
                            *left = Some(node_idx);
                        }
                    }
                }
            }

            Ok(Tree {
                root,
                nodes: nodes
                    .into_iter()
                    .map(TryInto::try_into)
                    .collect::<Result<Vec<Node>, InvalidBitmap>>()?,
                symbol_bitmap,
            })
        }
    }

    /// <https://en.wikipedia.org/wiki/Canonical_Huffman_code#Pseudocode>
    fn canonical_huffman_table(symbol_lengths: &[u8]) -> SymbolBitMap {
        debug_assert!(
            symbol_lengths.len() >= 1,
            "symbol_lengths must have at least one item"
        );
        let mut sorted_values = symbol_lengths.to_owned();
        sorted_values.sort_by_key(|&v| Reverse(v));
        let mut code = 0;
        // This maps a length to a list of codes that go with that length
        let mut code_map: HashMap<u8, Vec<Vec<bool>>> = HashMap::new();

        let mut len = sorted_values.pop().expect("This must be at least one");
        code_map.insert(len, vec![vec![false; len.into()]]);

        for sorted_value in sorted_values.into_iter().rev() {
            code = (code + 1) << (sorted_value - len);

            let entry = code_map.entry(sorted_value).or_default();

            entry.push(code_to_bitvec(code, sorted_value));

            len = sorted_value;
        }

        let mut symbol_bit_map = SymbolBitMap::default();

        for (symbol, length) in symbol_lengths.iter().enumerate() {
            let entry = code_map.get_mut(length).expect("We just built this");

            let symbol = match symbol {
                0 => Symbol::RunA,
                1 => Symbol::RunB,
                idx if idx < symbol_lengths.len() - 1 => Symbol::Byte((idx - 1) as u16),
                _ => Symbol::Eob,
            };

            symbol_bit_map.insert(symbol, entry.remove(0));
        }

        symbol_bit_map
    }

    /// Convert a huffman code to a bitvec of the given length
    fn code_to_bitvec(mut code: u64, len: u8) -> Vec<bool> {
        let mut bitmap = vec![];

        for _ in 0..len {
            if code % 2 == 0 {
                bitmap.push(false);
            } else {
                bitmap.push(true);
            }

            code >>= 1;
        }
        bitmap.reverse();

        bitmap
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        mod test_try_from_vec {
            use super::*;

            #[test]
            fn simple() {
                let symbol_bitmap: SymbolBitMap =
                    [(Symbol::RunA, vec![false]), (Symbol::RunB, vec![true])]
                        .into_iter()
                        .collect();
                let tree: Tree = symbol_bitmap.try_into().unwrap();

                assert_eq!(tree.nodes.len(), 3);
                match tree.nodes[tree.root.0] {
                    Node::Branch(left, right) => {
                        let left = &tree.nodes[left.0];
                        let right = &tree.nodes[right.0];
                        match (left, right) {
                            (Node::Leaf(Symbol::RunA), Node::Leaf(Symbol::RunB)) => {}
                            _ => panic!("nodes should be the correct symbols"),
                        }
                    }
                    _ => panic!("root should be a branch"),
                }
            }

            #[test]
            fn example_1() {
                let symbol_lengths = vec![
                    2, 5, 4, 5, 6, 5, 5, 4, 9, 5, 5, 5, 4, 5, 4, 5, 9, 4, 8, 5, 4, 5, 8, 8,
                ];

                let tree: Tree = symbol_lengths.try_into().unwrap();

                assert_eq!(
                    tree.decode(&mut Bitstream::new(&[0_u8][..])).unwrap(),
                    Symbol::RunA
                );
                assert_eq!(
                    tree.decode(&mut Bitstream::new(&[160_u8][..])).unwrap(),
                    Symbol::RunB
                );
                assert_eq!(
                    tree.decode(&mut Bitstream::new(&[176_u8][..])).unwrap(),
                    Symbol::Byte(4)
                );
                assert_eq!(
                    tree.decode(&mut Bitstream::new(&[255_u8, 128_u8][..]))
                        .unwrap(),
                    Symbol::Byte(15)
                );
                assert_eq!(
                    tree.decode(&mut Bitstream::new(&[252_u8][..])).unwrap(),
                    Symbol::Byte(17)
                );
                assert_eq!(
                    tree.decode(&mut Bitstream::new(&[254_u8][..])).unwrap(),
                    Symbol::Eob
                );
            }
        }

        /// Test [`canonical_huffman_table`].
        mod test_canonical_huffman_table {
            use super::*;

            /// Use an example from A.2.3 in the PDF
            #[test]
            fn example_1() {
                let symbol_lengths = [
                    2, 5, 4, 5, 6, 5, 5, 4, 9, 5, 5, 5, 4, 5, 4, 5, 9, 4, 8, 5, 4, 5, 8, 8,
                ];

                let table = canonical_huffman_table(&symbol_lengths);

                assert_eq!(table.get(&Symbol::RunA).unwrap(), &[false, false]);
                assert_eq!(
                    table.get(&Symbol::RunB).unwrap(),
                    &[true, false, true, false, false]
                );
                assert_eq!(
                    table.get(&Symbol::Byte(4)).unwrap(),
                    &[true, false, true, true, false]
                );
                assert_eq!(table.get(&Symbol::Byte(15)).unwrap(), &[true; 9]);
                assert_eq!(
                    table.get(&Symbol::Byte(17)).unwrap(),
                    &[true, true, true, true, true, true, false, false]
                );
                assert_eq!(
                    table.get(&Symbol::Eob).unwrap(),
                    &[true, true, true, true, true, true, true, false]
                );
            }

            /// TODONEXT: Encode the other example as a test
            #[test]
            fn example_2() {
                let symbol_lengths = [
                    1, 4, 5, 5, 6, 10, 4, 10, 10, 5, 10, 10, 10, 10, 4, 10, 9, 9, 9, 4, 9, 5, 4, 5,
                ];

                let table = canonical_huffman_table(&symbol_lengths);

                assert_eq!(table.get(&Symbol::RunA).unwrap(), &[false]);
                assert_eq!(
                    table.get(&Symbol::RunB).unwrap(),
                    &[true, false, false, false]
                );
                assert_eq!(
                    table.get(&Symbol::Byte(4)).unwrap(),
                    &[true, true, true, true, true, true, true, false, false, false]
                );
                assert_eq!(
                    table.get(&Symbol::Byte(15)).unwrap(),
                    &[true, true, true, true, true, true, false, false, false,]
                );
                assert_eq!(
                    table.get(&Symbol::Byte(17)).unwrap(),
                    &[true, true, true, true, true, true, false, true, false]
                );
                assert_eq!(
                    table.get(&Symbol::Eob).unwrap(),
                    &[true, true, true, true, false]
                );
            }
        }

        /// Test [`code_to_bitvec`].
        mod test_code_to_bitvec {
            use super::*;

            /// Convert code 0 of length 2
            #[test]
            fn code_0_2() {
                let bv = code_to_bitvec(0, 2);

                assert_eq!(bv, [false, false]);
            }

            /// Convert code 4 of length 4
            #[test]
            fn code_4_4() {
                let bv = code_to_bitvec(4, 4);

                assert_eq!(bv, [false, true, false, false]);
            }

            /// Convert code 20 of length 5
            #[test]
            fn code_20_5() {
                let bv = code_to_bitvec(20, 5);

                assert_eq!(bv, [true, false, true, false, false]);
            }

            /// Convert code 21 of length 5
            #[test]
            fn code_21_5() {
                let bv = code_to_bitvec(21, 5);

                assert_eq!(bv, [true, false, true, false, true]);
            }

            /// Convert code 511 of length 9
            #[test]
            fn code_511_9() {
                let bv = code_to_bitvec(511, 9);

                assert_eq!(bv, [true; 9]);
            }
        }
    }
}

/// Encode RLE `Symbol`s into Huffman tables.
pub(super) fn encode(data: &[rle2::Symbol]) -> HuffmanCodedData {
    let mut huffman_symbols: Vec<Symbol> = Vec::new();
    let mut symbol_weights: HashMap<Symbol, usize> = HashMap::new();
    for symbol in data {
        let symbol = symbol.into();
        huffman_symbols.push(symbol);
        symbol_weights
            .entry(symbol)
            .and_modify(|v| *v += 1)
            .or_insert(1);
    }
    huffman_symbols.push(Symbol::Eob);
    symbol_weights.insert(Symbol::Eob, 1);

    let tree = tree::Tree::new(symbol_weights);

    let bits = tree.encode(data);
    let block = HuffmanBlock {
        tree_index: 0,
        bitvec: bits,
    };

    HuffmanCodedData {
        trees: vec![tree],
        blocks: vec![block],
    }
}

/// Decode the `Symbol`s back to bytes.
pub(super) fn decode(data: &[Symbol]) -> Vec<rle2::Symbol> {
    data.iter()
        .filter_map(|symbol| {
            if *symbol != Symbol::Eob {
                Some(symbol.into())
            } else {
                None
            }
        })
        .collect()
}
