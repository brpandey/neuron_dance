use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use ndarray::Array2;
use std::fs::File;
use std::io::{Cursor, Read};

use crate::dataset::DATASET_DIR;
use crate::dataset::{DataSet, TrainTestSplitData};

// MnistData

type Subsets = (SubsetType, SubsetType, SubsetType, SubsetType);
type RawQuad = (Raw, Raw, Raw, Raw); // (x_train, y_train, x_test, y_test)

pub struct MnistData(MnistType, Subsets, Option<Box<RawQuad>>);

impl MnistData {
    pub fn new(mtype: MnistType) -> Self {
        let subsets = (
            SubsetType::Train(RawType::Images), SubsetType::Train(RawType::Labels), // train
            SubsetType::Test(RawType::Images), SubsetType::Test(RawType::Labels) // test
        );

        MnistData(mtype, subsets, None)
    }
}

impl DataSet for MnistData {
    fn fetch(&mut self, token: &str) {
        let x_raw = self.1.0.fetch(token);
        let y_raw = self.1.1.fetch(token);
        let x_test = self.1.2.fetch(token);
        let y_test = self.1.3.fetch(token);
        self.2 = Some(Box::new((x_raw, y_raw, x_test, y_test)))
    }

    fn train_test_split(&mut self, _split_ratio: f32) -> TrainTestSplitData {
        if self.2.is_none() {
            let token = self.0.token();
            self.fetch(&token);
        }

        // Extract data from boxed raws
        let boxed_raws = self.2.as_mut().unwrap();

        let x_train = boxed_raws.0.take().unwrap();
        let y_train = boxed_raws.1.take().unwrap();
        let x_test = boxed_raws.2.take().unwrap();
        let y_test = boxed_raws.3.take().unwrap();

        let n_train = boxed_raws.0.size();
        let n_test = boxed_raws.2.size();

        self.2 = None;

        println!("x_train shape is {:?}, y_train shape is {:?}, x_test shape is {:?}, y_test shape is {:?}",
                 x_train.shape(), y_train.shape(), x_test.shape(), y_test.shape());

        TrainTestSplitData(x_train, y_train, n_train, x_test, y_test, n_test)
    }
}

/*****************************************************************/

// Start of Type Data

// 1 MnistType

pub enum MnistType {
    Regular,
    Fashion,
}

impl MnistType {
    pub fn token(&self) -> String {
        match self {
            MnistType::Regular => String::from("mnist"),
            MnistType::Fashion => String::from("mnist-fashion"),
        }
    }
}


// 2 SubsetType

// define types via enums to aid in constructing actual data enums
#[derive(Copy, Clone, PartialEq)]
enum SubsetType {
    Train(RawType),
    Test(RawType),
}

impl SubsetType {
    fn path(&self, type_dir: &str) -> String {
        match self {
            SubsetType::Train(ref r) => self.merge_path(type_dir, "train", r.filename()),
            SubsetType::Test(ref r) => self.merge_path(type_dir, "t10k", r.filename()),
        }
    }

    fn merge_path(&self, type_dir: &str, token: &str, suffix_path: &str) -> String {
        format!("{}{}/{}-{}", DATASET_DIR, type_dir, token, suffix_path)
    }

    fn fetch(&self, type_dir_name: &str) -> Raw {
        let path = self.path(type_dir_name);

        let f = File::open(path).unwrap();
        let mut decoder = GzDecoder::new(f);

        // decode entire gzip file into buf
        let mut buf: Vec<u8> = vec![];
        decoder.read_to_end(&mut buf).unwrap();

        // Create content from decoded file
        Raw::new(buf)
    }
}

// 3 RawType

#[derive(Copy, Clone, PartialEq)]
enum RawType { Labels, Images }

impl RawType {
    fn filename(&self) -> &str {
        match self {
            RawType::Images  => "images-idx3-ubyte.gz",
            RawType::Labels  => "labels-idx1-ubyte.gz",
        }
    }
}

/************************** Raw Data Types *******************************/

pub enum Raw {
    Labels(RawLabels),
    Images(RawImages),
}

impl Raw {
    const IDX_MAGIC_BYTES_LEN: usize = 4;

    fn new(buf: Vec<u8>) -> Self {
        // wrap the in-memory buffer with Cursor, which implements the Read trait
        let mut cur = Cursor::new(buf);
        let mut idx_magic_buf = [0u8; Self::IDX_MAGIC_BYTES_LEN];
        cur.read_exact(&mut idx_magic_buf).unwrap();

        let num_dim = idx_magic_buf[3] as usize; // byte 4 -- num dimensions in idx format
        let magic = u32::from_be_bytes(idx_magic_buf); // reconstitute back into (unsigned) 4 byte integer

        match magic {
            RawLabels::MAGIC => Raw::Labels(RawLabels::new(&mut cur, num_dim)),
            RawImages::MAGIC => Raw::Images(RawImages::new(&mut cur, num_dim)),
            0_u32..=2048_u32 | 2050_u32 | 2052_u32..=u32::MAX => todo!(),
        }
    }

    fn take(&mut self) -> Option<Array2<f64>> {
        match self {
            Raw::Labels(r) => r.take(),
            Raw::Images(r) => r.take(),
        }
    }

    fn size(&self) -> usize {
        match self {
            Raw::Labels(r) => r.1,
            Raw::Images(r) => r.1,
        }
    }
}

pub struct RawLabels(Option<Array2<f64>>, usize);
pub struct RawImages(Option<Array2<f64>>, usize, usize, usize);

impl RawLabels {
    const MAGIC: u32 = 2049;

    fn new(cur: &mut Cursor<Vec<u8>>, _ndim: usize) -> Self {
        let n_labels = cur.read_u32::<BigEndian>().unwrap();

        // read data into buf after metadata is read
        let mut buf: Vec<u8> = vec![];
        cur.read_to_end(&mut buf).unwrap();

        let floats: Vec<f64> = buf.iter().map(|ch| *ch as f64).collect();
        let data = Array2::from_shape_vec((n_labels as usize, 1), floats).unwrap(); // e.g. 10,000 x 1
        Self(Some(data), n_labels as usize)
    }

    fn take(&mut self) -> Option<Array2<f64>> { self.0.take() }
}

impl RawImages {
    const MAGIC: u32 = 2051;

    fn new(cur: &mut Cursor<Vec<u8>>, ndim: usize) -> Self {
        // extract sizes info given magic number metadata info about num dimensions
        let sizes = (0..ndim).into_iter().fold(Vec::with_capacity(ndim), |mut acc, _| {
            acc.push(cur.read_u32::<BigEndian>().unwrap() as usize); acc
        });

        let (n_images, shape1, shape2) = (sizes[0], sizes[1], sizes[2]);

        // read data into buf after metadata e.g. size, shapes is read
        let mut buf: Vec<u8> = vec![];
        cur.read_to_end(&mut buf).unwrap();

        let flattened_shape = shape1 * shape2 as usize; // instead of a 28x28 matrix, we grab 784 * 1 in an array2
        let floats: Vec<f64> = buf.iter().map(|ch| *ch as f64 / 255.0 as f64).collect(); // normalize to value between 0 and 1
        let data = Array2::from_shape_vec((n_images, flattened_shape), floats).unwrap(); // e.g. 10,000 x 784
        Self(Some(data), n_images, shape1, shape2)
    }

    fn take(&mut self) -> Option<Array2<f64>> { self.0.take() }
}
