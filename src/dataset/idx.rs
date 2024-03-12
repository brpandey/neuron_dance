use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use ndarray::Array2;
use std::fs::File;
use std::io::{Cursor, Read};

use crate::dataset::{DATASET_DIR, DataSet, TrainTestTuple, TrainTestSplitData};

// MnistData
type Subsets = (Subset, Subset, Subset, Subset);
type RawQuad = (Raw, Raw, Raw, Raw); // (x_train, y_train, x_test, y_test)

pub struct MnistData(MnistType, Subsets, Option<Box<RawQuad>>);

impl MnistData {
    pub fn new(mtype: MnistType) -> Self {
        let subsets = (
            Subset::Train(Raw::Images(None)), Subset::Train(Raw::Labels(None)), // train
            Subset::Test(Raw::Images(None)), Subset::Test(Raw::Labels(None)) // test
        );

        MnistData(mtype, subsets, None)
    }

    pub fn destructure(&mut self) -> TrainTestTuple {
        let q = self.2.as_mut().unwrap(); // Grab option as_mut, get raw quads

        let (n_train, n_test) = (q.0.size(), q.2.size());

        let ttt = // train test tuple
            (q.0.take().unwrap(), q.1.take().unwrap(), n_train,
             q.2.take().unwrap(), q.3.take().unwrap(), n_test);

        self.2 = None;

        ttt
    }
}

impl DataSet for MnistData {
    fn fetch(&mut self, t: &str) {
        // only fetch if data not resident already
        if self.2.is_none() {
            let x = self.1.0.fetch(t);
            let y = self.1.1.fetch(t);
            let x_test = self.1.2.fetch(t);
            let y_test = self.1.3.fetch(t);

            self.2 = Some(Box::new((x, y, x_test, y_test)));
        }
    }

    fn train_test_split(&mut self, _split_ratio: f32) -> TrainTestSplitData {
        self.fetch(&self.0.token());

        // Extract data from boxed raws
        let tts = TrainTestSplitData(self.destructure());
        println!("Train test split shapes are {}", &tts);
        tts
    }
}

/*****************************************************************/

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

// 2 Subset

enum Subset {
    Train(Raw),
    Test(Raw),
}

impl Subset {
    fn path(&self, type_dir: &str) -> String {
        match self {
            Subset::Train(ref r) => self.merge_path(type_dir, "train", r.filename()),
            Subset::Test(ref r) => self.merge_path(type_dir, "t10k", r.filename()),
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
        let mut decoded_buf: Vec<u8> = vec![];
        decoder.read_to_end(&mut decoded_buf).unwrap();

        // Create content from decoded file
        Raw::new(decoded_buf)
    }
}


// 3 Raw Data Types

pub enum Raw {
    Labels(Option<RawLabels>),
    Images(Option<RawImages>),
}

impl Raw {
    const IDX_MAGIC_BYTES_LEN: usize = 4;

    fn new(decoded_buf: Vec<u8>) -> Self {
        // wrap the in-memory buffer with Cursor, which implements the Read trait
        let mut cur = Cursor::new(decoded_buf);
        let mut idx_magic_buf = [0u8; Self::IDX_MAGIC_BYTES_LEN];
        cur.read_exact(&mut idx_magic_buf).unwrap();

        let num_dim = idx_magic_buf[3] as usize; // byte 4 -- num dimensions in idx format
        let magic = u32::from_be_bytes(idx_magic_buf); // reconstitute back into (unsigned) 4 byte integer

        match magic {
            RawLabels::MAGIC => Raw::Labels(Some(RawLabels::new(&mut cur, num_dim))),
            RawImages::MAGIC => Raw::Images(Some(RawImages::new(&mut cur, num_dim))),
            0_u32..=2048_u32 | 2050_u32 | 2052_u32..=u32::MAX => todo!(),
        }
    }

    fn filename(&self) -> &str {
        match self {
            Raw::Labels(_) => "labels-idx1-ubyte.gz",
            Raw::Images(_) => "images-idx3-ubyte.gz",
        }
    }

    fn take(&mut self) -> Option<Array2<f64>> {
        match self {
            Raw::Labels(Some(ref mut r)) => r.take(),
            Raw::Images(Some(ref mut r)) => r.take(),
            _ => None,
        }
    }


    fn size(&self) -> usize {
        match self {
            Raw::Labels(Some(ref r)) => r.1,  // access size field for RawLabels
            Raw::Images(Some(ref r)) => r.1, // access size field for RawImages
            _ => 0,
        }
    }
}

pub struct RawLabels(Option<Array2<f64>>, usize);
pub struct RawImages(Option<Array2<f64>>, usize, usize, usize);

impl RawLabels {
    const MAGIC: u32 = 2049;

    fn new(cur: &mut Cursor<Vec<u8>>, _ndim: usize) -> Self {
        let n_labels = cur.read_u32::<BigEndian>().unwrap();

        // read data into buffer after metadata is read
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
            let size = cur.read_u32::<BigEndian>().unwrap() as usize;
            acc.push(size); acc
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
