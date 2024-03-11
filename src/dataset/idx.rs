use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use ndarray::Array2;
use std::fs::File;
use std::io::{Cursor, Read};

use crate::dataset::DATASET_DIR;
use crate::dataset::{DataSet, TrainTestSplitData};


/************************** MnistData *******************************/

pub struct MnistData(Mnist, IdxType);

impl MnistData {
    pub fn new(mtype: Mnist) -> Self {
        let idx_metadata = IdxType::new();
        MnistData(mtype, idx_metadata)
    }

    fn fetch(&self) -> (Raw, Raw, Raw, Raw) {
        let mnist_token = self.0.token();
        self.1.fetch(mnist_token)
    }
}

impl DataSet for MnistData {
    //impl DataSet for IdxType {
    fn train_test_split(&self, _split_ratio: f32) -> TrainTestSplitData {
        let (mut x_raw, mut y_raw, mut x_raw_test, mut y_raw_test) = self.fetch();

        let x_train = x_raw.take().unwrap();
        let y_train = y_raw.take().unwrap();
        let n_train = x_raw.size();

        let x_test = x_raw_test.take().unwrap();
        let y_test = y_raw_test.take().unwrap();
        let n_test = x_raw_test.size();

        println!("x_train shape is {:?}, y_train shape is {:?}, x_test shape is {:?}, y_test shape is {:?}", x_train.shape(), y_train.shape(), x_test.shape(), y_test.shape());

        TrainTestSplitData(x_train, y_train, n_train, x_test, y_test, n_test)
    }
}

/******************************/
/***** Start of Type data *****/
/******************************/


/************************** Mnist *******************************/

pub enum Mnist {
    Regular,
    Fashion,
}

impl Mnist {
    pub fn token(&self) -> &str {
        match self {
            Mnist::Regular => "mnist",
            Mnist::Fashion => "mnist-fashion",
        }
    }
}


/************************** IdxType *******************************/
// Idx is the file format the mnist files are stored in

// aggregate type (x_train, y_train, x_test, y_test)
pub struct IdxType(SubsetType, SubsetType, SubsetType, SubsetType);

impl IdxType {
    pub fn new() -> Self {
        IdxType(
            SubsetType::Train(RawType::Images), SubsetType::Train(RawType::Labels), // train
            SubsetType::Test(RawType::Images), SubsetType::Test(RawType::Labels) // test
        )
    }

    fn fetch(&self, token: &str) -> (Raw, Raw, Raw, Raw) {
        let x_raw = self.0.fetch(token);
        let y_raw = self.1.fetch(token);
        let x_test = self.2.fetch(token);
        let y_test = self.3.fetch(token);
        (x_raw, y_raw, x_test, y_test)
    }
}

/************************** SubsetType *******************************/

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

/************************** RawType *******************************/

#[derive(Copy, Clone, PartialEq)]
pub enum RawType { Labels, Images }

impl RawType {
    fn filename(&self) -> &str {
        match self {
            RawType::Images  => "images-idx3-ubyte.gz",
            RawType::Labels  => "labels-idx1-ubyte.gz",
        }
    }
}

/************************** Raw DATA types *******************************/

enum Raw {
    Labels(RawLabels),
    Images(RawImages),
}

impl Raw {
    fn new(buf: Vec<u8>) -> Self {
        // wrap cursor around the in-memory buffer, implementing the Read trait
        let mut cur = Cursor::new(buf);

        match cur.read_u32::<BigEndian>().unwrap() { // match on Idx Magic Number
            RawLabels::MAGIC => Raw::Labels(RawLabels::new(&mut cur)),
            RawImages::MAGIC => Raw::Images(RawImages::new(&mut cur)),
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

struct RawLabels(Option<Array2<f64>>, usize);
struct RawImages(Option<Array2<f64>>, usize, usize, usize);

impl RawLabels {
    const MAGIC: u32 = 2049;

    fn new(cur: &mut Cursor<Vec<u8>>) -> Self {
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

    fn new(cur: &mut Cursor<Vec<u8>>) -> Self {
        let n_images = cur.read_u32::<BigEndian>().unwrap();
        let shape1 = cur.read_u32::<BigEndian>().unwrap(); // e.g. 28
        let shape2 = cur.read_u32::<BigEndian>().unwrap(); // e.g. 28

        // read data into buf after metadata e.g. size, shapes is read
        let mut buf: Vec<u8> = vec![];
        cur.read_to_end(&mut buf).unwrap();

        let flattened_shape = shape1 * shape2; // instead of a 28x28 matrix, we grab 784 * 1 in an array2
        let floats: Vec<f64> = buf.iter().map(|ch| *ch as f64 / 255.0 as f64).collect();

        let data = Array2::from_shape_vec((n_images as usize, flattened_shape as usize), floats).unwrap(); // e.g. 10,000 x 784
        Self(Some(data), n_images as usize, shape1 as usize, shape2 as usize)
    }

    fn take(&mut self) -> Option<Array2<f64>> { self.0.take() }
}
