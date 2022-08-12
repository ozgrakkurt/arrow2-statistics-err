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

fn block_schema() -> Schema {
    Schema::from(vec![
        Field::new("number", DataType::Int64, false),
        Field::new("nonce", DataType::UInt64, false),
        Field::new("hash", DataType::Binary, false),
    ])
}

fn options() -> WriteOptions {
    WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Snappy,
        version: Version::V2,
    }
}

#[derive(Debug, Default)]
pub struct Blocks {
    pub number: Int64Vec,
    pub nonce: UInt64Vec,
    pub hash: MutableBinaryArray,
    pub len: usize,
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
            arrow_take(self.nonce.as_box().as_ref(), &indices).unwrap(),
        ])
    }

    fn push(&mut self, elem: Self::Elem) -> Result<(), ()> {
        self.number.push(Some(elem.number));
        self.hash.push(Some(elem.hash.as_slice()));
        self.nonce.push(Some(elem.nonce));

        self.len += 1;

        Ok(())
    }

    fn len(&self) -> usize {
        self.len
    }

    fn schema() -> Schema {
        block_schema()
    }
}

pub trait IntoRowGroups: Default + std::marker::Sized + Send + Sync {
    type Elem: Send + Sync + std::fmt::Debug + 'static + std::marker::Sized;

    fn schema() -> Schema;
    fn into_chunk(self) -> Chunk;
    fn into_row_groups(elems: Vec<Self>) -> (RowGroups, Schema, WriteOptions) {
        let schema = Self::schema();

        let encoding_map = |_data_type: &DataType| Encoding::Plain;

        let encodings = (&schema.fields)
            .iter()
            .map(|f| transverse(&f.data_type, encoding_map))
            .collect::<Vec<_>>();

        let row_groups = RowGroupIterator::try_new(
            elems
                .into_par_iter()
                .map(|elem| Ok(Self::into_chunk(elem)))
                .collect::<Vec<_>>()
                .into_iter(),
            &schema,
            options(),
            encodings,
        )
        .unwrap();

        (row_groups, schema, options())
    }
    fn push(&mut self, elem: Self::Elem) -> Result<(), ()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Default)]
pub struct Block {
    pub number: i64,
    pub hash: Vec<u8>,
    pub nonce: u64,
}
