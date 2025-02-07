use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use std::{
    fs::File,
    io::{Cursor, Read},
};

use crate::dataset::{DataSet, DataSetFormat, TrainTestSubsets, TrainTestTuple, ROOT_DIR};
use crate::{types::SimpleError, visualize::Peek};

// MnistData
type Subsets = (Subset, Subset, Subset, Subset);

pub struct MnistData {
    mtype: MnistType,
    subset_types: Subsets,
    data: Option<TrainTestTuple>,
    class_names: Option<Vec<String>>,
}

impl MnistData {
    pub const SHAPE: (usize, usize) = (28, 28);
    pub const N_FEATURES: usize = 784;

    pub fn new(mtype: MnistType) -> Self {
        let subset_types = (
            Subset::Train(Raw::Images(None)),
            Subset::Train(Raw::Labels(None)), // train
            Subset::Test(Raw::Images(None)),
            Subset::Test(Raw::Labels(None)), // test
        );

        MnistData {
            mtype,
            subset_types,
            data: None,
            class_names: mtype.class_names(),
        }
    }
}

impl Peek for MnistData {
    fn peek<S: Data<Elem = f64>>(x: &ArrayBase<S, Ix2>, text: Option<&str>) {
        use crate::pool::{Pool, PoolType};
        use crate::visualize::{Empty, Visualize};

        let revert = x.dim(); // stash original shape
        let image = x.to_shape(Self::SHAPE).unwrap();

        if let Some(reduced_image) = Pool::apply(&image, 2, 2, PoolType::Max) {
            Visualize::table_preview(&reduced_image, None::<Empty>, true, text);
        }

        x.to_shape(revert).unwrap(); // revert back
    }
}

impl DataSet for MnistData {
    fn fetch(&mut self) -> Result<(), SimpleError> {
        // only fetch if data not resident already
        if self.data.is_none() {
            let t = &self.mtype.token();

            let (mut x_raw, mut y_raw) =
                (self.subset_types.0.fetch(t)?, self.subset_types.1.fetch(t)?);
            let (mut x_raw_test, mut y_raw_test) =
                (self.subset_types.2.fetch(t)?, self.subset_types.3.fetch(t)?);

            let ttt = // train test tuple
                (x_raw.take().unwrap(), y_raw.take().unwrap(), x_raw.size(),
                 x_raw_test.take().unwrap(), y_raw_test.take().unwrap(), x_raw_test.size(), MnistData::N_FEATURES);

            self.data = Some(ttt);
        }

        Ok(())
    }

    fn head(&self) {
        use crate::visualize::Visualize;
        let num_heatmaps = 7;

        if self.data.is_none() {
            return;
        } // if data hasn't been fetched, return early

        println!("> head 0..{num_heatmaps} mnist-file | heatmap");

        let x_train = &self.data.as_ref().unwrap().0;

        for i in 0..num_heatmaps {
            let x_row = x_train.row(i);
            let image_view = x_row.into_shape_with_order(Self::SHAPE).unwrap();
            Visualize::heatmap_row(&image_view, i as u8);
        }
    }

    fn shuffle(&mut self) {}

    fn train_test_split(&mut self, _split_ratio: f32) -> TrainTestSubsets {
        let _ = self.fetch();

        // Extract data from boxed raws
        let tts = TrainTestSubsets::new(
            DataSetFormat::IDX,
            self.data.take().unwrap(),
            self.class_names.clone(),
        );

        println!("Loaded data, subset shapes are {}\n", &tts);
        tts
    }
}

/*****************************************************************/

// 1 MnistType

#[derive(Copy, Clone)]
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

    pub fn class_names(&self) -> Option<Vec<String>> {
        match self {
            MnistType::Regular => None,
            MnistType::Fashion => {
                let n = vec![
                    "T-shirt",
                    "Trouser",
                    "Pullover",
                    "Dress",
                    "Coat",
                    "Sandal",
                    "Shirt",
                    "Sneaker",
                    "Bag",
                    "Ankle boot",
                ];
                let names = n.into_iter().map(|v| v.to_string()).collect();
                Some(names)
            }
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
        format!("{}/data/{}/{}-{}", ROOT_DIR, type_dir, token, suffix_path)
    }

    fn fetch(&self, type_dir_name: &str) -> Result<Raw, SimpleError> {
        let path = self.path(type_dir_name);

        let f = File::open(path)?;
        let mut decoder = GzDecoder::new(f);

        // decode entire gzip file into buf
        let mut decoded_buf: Vec<u8> = vec![];
        decoder.read_to_end(&mut decoded_buf)?;

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

    fn new(decoded_buf: Vec<u8>) -> Result<Self, SimpleError> {
        // wrap the in-memory buffer with Cursor, which implements the Read trait
        let mut cur = Cursor::new(decoded_buf);
        let mut idx_magic_buf = [0u8; Self::IDX_MAGIC_BYTES_LEN];
        cur.read_exact(&mut idx_magic_buf)?;

        let num_dim = idx_magic_buf[3] as usize; // byte 4 -- num dimensions in idx format
        let magic = u32::from_be_bytes(idx_magic_buf); // reconstitute back into (unsigned) 4 byte integer

        let raw = match magic {
            RawLabels::MAGIC => Raw::Labels(Some(RawLabels::new(&mut cur, num_dim)?)),
            RawImages::MAGIC => Raw::Images(Some(RawImages::new(&mut cur, num_dim)?)),
            0_u32..=2048_u32 | 2050_u32 | 2052_u32..=u32::MAX => todo!(),
        };

        Ok(raw)
    }

    fn filename(&self) -> &str {
        match self {
            Raw::Labels(_) => "labels-idx1-ubyte.gz",
            Raw::Images(_) => "images-idx3-ubyte.gz",
        }
    }

    fn take(&mut self) -> Option<Array2<f64>> {
        match self {
            Raw::Labels(Some(ref mut r)) => r.0.take(),
            Raw::Images(Some(ref mut r)) => r.0.take(),
            _ => None,
        }
    }

    fn size(&self) -> usize {
        match self {
            Raw::Labels(Some(ref r)) => r.1, // access size field for RawLabels
            Raw::Images(Some(ref r)) => r.1, // access size field for RawImages
            _ => 0,
        }
    }
}

pub struct RawLabels(Option<Array2<f64>>, usize);

impl RawLabels {
    const MAGIC: u32 = 2049;

    fn new(cur: &mut Cursor<Vec<u8>>, _ndim: usize) -> Result<Self, SimpleError> {
        let n_labels = cur.read_u32::<BigEndian>()?;

        // read data into buffer after metadata is read
        let mut buf: Vec<u8> = vec![];
        cur.read_to_end(&mut buf)?;

        let floats: Vec<f64> = buf.iter().map(|ch| *ch as f64).collect();
        let data = Array2::from_shape_vec((n_labels as usize, 1), floats)?; // e.g. 10,000 x 1
        Ok(Self(Some(data), n_labels as usize))
    }
}

pub struct RawImages(Option<Array2<f64>>, usize); //, usize, usize);

impl RawImages {
    const MAGIC: u32 = 2051;

    fn new(cur: &mut Cursor<Vec<u8>>, ndim: usize) -> Result<Self, SimpleError> {
        // extract sizes info given magic number metadata info about num dimensions
        let acc: Result<Vec<usize>, SimpleError> = Ok(Vec::with_capacity(ndim));
        let sizes = (0..ndim).fold(acc, |mut acc, _| {
            let size = cur.read_u32::<BigEndian>()?;
            acc.as_mut().unwrap().push(size as usize);
            acc
        })?;

        let (n_images, shape1, shape2) = (sizes[0], sizes[1], sizes[2]);

        // read data into buf after metadata e.g. size, shapes is read
        let mut buf: Vec<u8> = vec![];
        cur.read_to_end(&mut buf)?;

        let flattened_shape = shape1 * shape2; // instead of a 28x28 matrix, we grab 784 * 1 in an array2
        assert_eq!(flattened_shape, MnistData::N_FEATURES);

        let floats: Vec<f64> = buf.iter().map(|ch| *ch as f64 / 255.0).collect(); // normalize to value between 0 and 1
        let data = Array2::from_shape_vec((n_images, flattened_shape), floats)?; // e.g. 10,000 x 784
        Ok(Self(Some(data), n_images))
    }
}
