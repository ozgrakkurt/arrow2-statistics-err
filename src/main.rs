use arrow2::io::parquet::write::*;
use std::path::Path;
use std::time::Instant;
use std::{cmp, fs, mem};

use arrow2::array::{
    Array, Int64Vec, MutableArray, MutableBinaryArray as ArrowMutableBinaryArray,
    MutableBooleanArray, UInt64Vec,
};
use arrow2::chunk::Chunk as ArrowChunk;
use arrow2::compute::sort::{lexsort_to_indices, sort_to_indices, SortColumn, SortOptions};
use arrow2::compute::take::take as arrow_take;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::error::Error as ArrowError;
use arrow2::io::parquet::write::{
    CompressionOptions, Encoding, RowGroupIterator, Version, WriteOptions,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::result::Result as StdResult;

type Chunk = ArrowChunk<Box<dyn Array>>;
type MutableBinaryArray = ArrowMutableBinaryArray<i64>;

fn main() {
    let mut blocks = Blocks::default();

    for _ in 0..10000 {
        blocks.push(Block::default()).unwrap();
    }

    let (row_groups, schema, options) = Blocks::into_row_groups(vec![blocks]);

    let file = fs::File::create("my_file.parquet").unwrap();
    let mut writer = FileWriter::try_new(file, schema, options).unwrap();

    for group in row_groups {
        writer.write(group.unwrap()).unwrap();
    }
    writer.end(None).unwrap();
}

fn bytes32() -> DataType {
    DataType::Binary
}

fn bloom_filter_bytes() -> DataType {
    DataType::Binary
}

fn address() -> DataType {
    DataType::Binary
}

fn bytes32_arr() -> MutableBinaryArray {
    MutableBinaryArray::default()
}

fn bloom_filter_arr() -> MutableBinaryArray {
    MutableBinaryArray::default()
}

fn address_arr() -> MutableBinaryArray {
    MutableBinaryArray::default()
}

fn block_schema() -> Schema {
    Schema::from(vec![
        Field::new("number", DataType::Int64, false),
        Field::new("hash", bytes32(), false),
        Field::new("parent_hash", bytes32(), false),
        Field::new("nonce", DataType::UInt64, false),
        Field::new("sha3_uncles", bytes32(), false),
        Field::new("logs_bloom", bloom_filter_bytes(), false),
        Field::new("transactions_root", bytes32(), false),
        Field::new("state_root", bytes32(), false),
        Field::new("receipts_root", bytes32(), false),
        Field::new("miner", address(), false),
        Field::new("difficulty", DataType::Binary, false),
        Field::new("total_difficulty", DataType::Binary, false),
        Field::new("extra_data", DataType::Binary, false),
        Field::new("size", DataType::Int64, false),
        Field::new("gas_limit", DataType::Binary, false),
        Field::new("gas_used", DataType::Binary, false),
        Field::new("timestamp", DataType::Int64, false),
    ])
}

fn options() -> WriteOptions {
    WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Snappy,
        version: Version::V2,
    }
}

#[derive(Debug)]
pub struct Blocks {
    pub number: Int64Vec,
    pub hash: MutableBinaryArray,
    pub parent_hash: MutableBinaryArray,
    pub nonce: UInt64Vec,
    pub sha3_uncles: MutableBinaryArray,
    pub logs_bloom: MutableBinaryArray,
    pub transactions_root: MutableBinaryArray,
    pub state_root: MutableBinaryArray,
    pub receipts_root: MutableBinaryArray,
    pub miner: MutableBinaryArray,
    pub difficulty: MutableBinaryArray,
    pub total_difficulty: MutableBinaryArray,
    pub extra_data: MutableBinaryArray,
    pub size: Int64Vec,
    pub gas_limit: MutableBinaryArray,
    pub gas_used: MutableBinaryArray,
    pub timestamp: Int64Vec,
    pub len: usize,
}

impl Default for Blocks {
    fn default() -> Self {
        Self {
            number: Default::default(),
            hash: bytes32_arr(),
            parent_hash: bytes32_arr(),
            nonce: Default::default(),
            sha3_uncles: bytes32_arr(),
            logs_bloom: bloom_filter_arr(),
            transactions_root: bytes32_arr(),
            state_root: bytes32_arr(),
            receipts_root: bytes32_arr(),
            miner: address_arr(),
            difficulty: Default::default(),
            total_difficulty: Default::default(),
            extra_data: Default::default(),
            size: Default::default(),
            gas_limit: Default::default(),
            gas_used: Default::default(),
            timestamp: Default::default(),
            len: 0,
        }
    }
}

type RowGroups = RowGroupIterator<Box<dyn Array>, std::vec::IntoIter<StdResult<Chunk, ArrowError>>>;

impl IntoRowGroups for Blocks {
    type Elem = Block;

    fn into_chunk(mut self) -> Chunk {
        let number = self.number.as_box();

        let indices = sort_to_indices::<i64>(
            number.as_ref(),
            &SortOptions {
                descending: false,
                nulls_first: true,
            },
            None,
        )
        .unwrap();

        Chunk::new(vec![
            arrow_take(number.as_ref(), &indices).unwrap(),
            arrow_take(self.hash.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.parent_hash.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.nonce.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.timestamp.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.sha3_uncles.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.logs_bloom.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.transactions_root.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.state_root.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.receipts_root.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.miner.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.difficulty.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.total_difficulty.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.extra_data.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.size.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.gas_limit.as_box().as_ref(), &indices).unwrap(),
            arrow_take(self.gas_used.as_box().as_ref(), &indices).unwrap(),
        ])
    }

    fn push(&mut self, elem: Self::Elem) -> Result<(), ()> {
        self.number.push(Some(elem.number));
        self.hash.push(Some(elem.hash.as_slice()));
        self.parent_hash.push(Some(elem.parent_hash.as_slice()));
        self.nonce.push(Some(elem.nonce));
        self.sha3_uncles.push(Some(elem.sha3_uncles.as_slice()));
        self.logs_bloom.push(Some(elem.logs_bloom.as_slice()));
        self.transactions_root
            .push(Some(elem.transactions_root.as_slice()));
        self.state_root.push(Some(elem.state_root.as_slice()));
        self.receipts_root.push(Some(elem.receipts_root.as_slice()));
        self.miner.push(Some(elem.miner.as_slice()));
        self.difficulty.push(Some(elem.difficulty));
        self.total_difficulty.push(Some(elem.total_difficulty));
        self.extra_data.push(Some(elem.extra_data));
        self.size.push(Some(elem.size));
        self.gas_limit.push(Some(elem.gas_limit));
        self.gas_used.push(Some(elem.gas_used));
        self.timestamp.push(Some(elem.timestamp));

        self.len += 1;

        Ok(())
    }

    fn len(&self) -> usize {
        self.len
    }

    fn encoding() -> Vec<Encoding> {
        vec![
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
            Encoding::Plain,
        ]
    }

    fn schema() -> Schema {
        block_schema()
    }
}

pub trait IntoRowGroups: Default + std::marker::Sized + Send + Sync {
    type Elem: Send + Sync + std::fmt::Debug + 'static + std::marker::Sized + BlockNum;

    fn encoding() -> Vec<Encoding>;
    fn schema() -> Schema;
    fn into_chunk(self) -> Chunk;
    fn into_row_groups(elems: Vec<Self>) -> (RowGroups, Schema, WriteOptions) {
        let row_groups = RowGroupIterator::try_new(
            elems
                .into_par_iter()
                .map(|elem| Ok(Self::into_chunk(elem)))
                .collect::<Vec<_>>()
                .into_iter(),
            &Self::schema(),
            options(),
            Self::encoding().into_iter().map(|e| vec![e]).collect(),
        )
        .unwrap();

        (row_groups, Self::schema(), options())
    }
    fn push(&mut self, elem: Self::Elem) -> Result<(), ()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait BlockNum {
    fn block_num(&self) -> usize;
}

impl BlockNum for Block {
    fn block_num(&self) -> usize {
        self.number as usize
    }
}

#[derive(Debug, Clone, Default)]
pub struct Block {
    pub number: i64,
    pub hash: Vec<u8>,
    pub parent_hash: Vec<u8>,
    pub nonce: u64,
    pub sha3_uncles: Vec<u8>,
    pub logs_bloom: Vec<u8>,
    pub transactions_root: Vec<u8>,
    pub state_root: Vec<u8>,
    pub receipts_root: Vec<u8>,
    pub miner: Vec<u8>,
    pub difficulty: Vec<u8>,
    pub total_difficulty: Vec<u8>,
    pub extra_data: Vec<u8>,
    pub size: i64,
    pub gas_limit: Vec<u8>,
    pub gas_used: Vec<u8>,
    pub timestamp: i64,
}
