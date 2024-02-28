//! Define the huffman encoding functions.

use std::collections::{BinaryHeap, HashMap};

use super::rle2;

#[derive(Debug)]
pub enum Error {
    InvalidNodeIndex,
    TruncatedBitstream,
}

#[derive(Debug, Default)]
pub(super) struct HuffmanCodedData {
    /// This is the set of trees used in the encoded data.
    trees: Vec<tree::Tree>,
    /// This contains the encoded data.
    blocks: Vec<HuffmanBlock>,
}

#[derive(Debug, Default)]
struct HuffmanBlock {
    /// Points into [`HuffmanCodedData::trees`] to indicate which tree was used for this block.
    tree_index: u8,
    /// This is the huffman-coded data.
    bitvec: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
enum Symbol {
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
            rle2::Symbol::Byte(byte) => Self::Byte(*byte as u16 + 1),
        }
    }
}

impl From<&Symbol> for rle2::Symbol {
    fn from(value: &Symbol) -> Self {
        match value {
            Symbol::RunA => rle2::Symbol::RunA,
            Symbol::RunB => rle2::Symbol::RunB,
            Symbol::Byte(byte) => rle2::Symbol::Byte((*byte - 1) as u8),
            // We'll strip off the EOB symbol before calling this function.
            Symbol::Eob => unreachable!(),
        }
    }
}

mod tree {
    use super::*;

    use std::cmp::Reverse;

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
    pub(super) struct Tree {
        root: NodeIdx,
        nodes: Vec<Node>,
        symbol_bitmap: SymbolBitMap,
    }

    impl Tree {
        pub(super) fn new(symbol_weights: HashMap<Symbol, usize>) -> Self {
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

        pub(super) fn encode(&self, symbols: &[rle2::Symbol]) -> Vec<u8> {
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

        pub(super) fn decode(&self, mut input: &[u8]) -> Result<Vec<Symbol>, Error> {
            let mut symbols = Vec::new();
            let mut location = self.root;

            while !input.is_empty() {
                let node = self.nodes.get(location.0).ok_or(Error::InvalidNodeIndex)?;
                match node {
                    Node::Leaf(symbol) => {
                        symbols.push(*symbol);
                        location = self.root;
                    }
                    Node::Branch(left, right) => {
                        let Some((first, rest)) = input.split_first() else {
                            unreachable!()
                        };
                        input = rest;
                        match *first {
                            0 => location = *left,
                            1 => location = *right,
                            // These values are really bits. Eventually, we'll make this impossible in the
                            // type system.
                            _ => unreachable!(),
                        }
                    }
                }
            }

            let node = self.nodes.get(location.0).ok_or(Error::InvalidNodeIndex)?;
            if let Node::Leaf(symbol) = node {
                symbols.push(*symbol);
            } else {
                return Err(Error::TruncatedBitstream);
            }

            Ok(symbols)
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
    dbg!(&tree);

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
pub(super) fn decode(data: &HuffmanCodedData) -> Result<Vec<rle2::Symbol>, Error> {
    let mut output = Vec::new();

    for block in &data.blocks {
        let tree = &data.trees[block.tree_index as usize];
        let mut symbols = tree.decode(block.bitvec.as_slice())?;
        output.append(&mut symbols);
    }

    Ok(output
        .iter()
        .filter_map(|symbol| {
            if *symbol != Symbol::Eob {
                Some(symbol.into())
            } else {
                None
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let input = &[
            rle2::Symbol::RunA,
            rle2::Symbol::Byte(2),
            rle2::Symbol::RunA,
            rle2::Symbol::Byte(0),
            rle2::Symbol::Byte(1),
            rle2::Symbol::Byte(2),
            rle2::Symbol::Byte(0),
            rle2::Symbol::Byte(2),
            rle2::Symbol::Byte(2),
            rle2::Symbol::RunB,
            rle2::Symbol::RunA,
            rle2::Symbol::RunB,
            rle2::Symbol::Byte(0),
            rle2::Symbol::Byte(2),
            rle2::Symbol::Byte(0),
        ];

        let output = decode(&encode(input)).unwrap();

        assert_eq!(input.as_slice(), output);
    }
}
