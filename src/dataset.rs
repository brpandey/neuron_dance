use ndarray::{Array2, Axis};
use std::{env, fmt};

use crate::algebra::AlgebraExt;

pub mod csv;
pub mod idx;

#[derive(Copy, Clone)]
pub enum DataSetFormat {
    CSV,
    IDX,
}

pub const ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");

pub trait DataSet {
    fn fetch(&mut self);
    fn head(&self);
    fn shuffle(&mut self);
    fn train_test_split(&mut self, split_ratio: f32) -> TrainTestSubsetData;
}

//                           x_train    y_train        # train  x_test     y_test     # test
pub type TrainTestTuple = (Array2<f64>, Array2<f64>, usize, Array2<f64>, Array2<f64>, usize);
pub struct TrainTestSubsetData{
    format: DataSetFormat,
    headers: Option<Vec<String>>,
    data: TrainTestTuple,
}

//                                   train         test
pub type TrainTestSubsetRef<'a> = (SubsetRef<'a>, SubsetRef<'a>);

impl TrainTestSubsetData {
    // scale is 0 min 1 max
    pub fn min_max_scale(&self, min: f64, max: f64) -> Self {
        let x_train = &self.data.0;
        let x_test = &self.data.3;

        // get the min and max values for each column
        let (x_train_mins, x_test_mins) = (x_train.min_axis(Axis(0)), x_test.min_axis(Axis(0)));
        let (x_train_maxs, x_test_maxs) = (x_train.max_axis(Axis(0)), x_test.max_axis(Axis(0)));

        let x_train_std = (x_train - &x_train_mins) / (&x_train_maxs - &x_train_mins);
        let x_test_std = (x_test - &x_test_mins) / (&x_test_maxs - &x_test_mins);

        let x_train_scaled = x_train_std*(max-min) + min;
        let x_test_scaled = x_test_std*(max-min) + min;

         TrainTestSubsetData {
             format: self.format.clone(),
             headers: self.headers.clone(),
             data: (x_train_scaled, self.data.1.clone(), self.data.2,
                    x_test_scaled, self.data.4.clone(), self.data.5)
        }
    }

    pub fn get_ref<'a>(&'a self) -> TrainTestSubsetRef<'a> {
        (
            SubsetRef {
                x: &self.data.0,
                y: &self.data.1,
                size: self.data.2,
                format: self.format.clone(),
            },
            SubsetRef {
                x: &self.data.3,
                y: &self.data.4,
                size: self.data.5,
                format: self.format.clone(),
            },
        )
    }
}

impl fmt::Display for TrainTestSubsetData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x_train shape {:?}, y_train shape  {:?}, x_test shape {:?}, y_test shape {:?}",
                 &self.data.0.shape(), &self.data.1.shape(), &self.data.3.shape(), &self.data.4.shape())
    }
}

#[derive(Copy, Clone)]
pub struct SubsetRef<'a> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array2<f64>,
    pub size: usize,
    pub format: DataSetFormat,
}

impl<'a> SubsetRef<'a> {
    pub fn random(&self) -> (Array2<f64>, Array2<f64>) {
        use rand::Rng;

        let (mut rng, random_index);
        rng = rand::thread_rng();
        random_index = rng.gen_range(0..self.size);

        let x_single = self.x.select(Axis(0), &[random_index]);
        let y_single = self.y.select(Axis(0), &[random_index]);

        (x_single, y_single)
    }

    pub fn peek(&self, x: &Array2<f64>) {
        use crate::dataset::{idx::MnistData, csv::CSVData};
        use crate::visualize::Peek;

        match self.format {
            DataSetFormat::CSV => CSVData::peek(x, Some("=> corresponding x input features, see tabular row")),
            DataSetFormat::IDX => MnistData::peek(x, Some("=> reduced x input image, see grid below")),
        }
    }
}
