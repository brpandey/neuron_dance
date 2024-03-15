use ndarray::Array2;
use std::fmt;

pub mod csv;
pub mod idx;

pub const DATASET_DIR: &str = "./src/dataset/raw/";

pub trait DataSet  {
    fn fetch(&mut self, token: &str);
    fn train_test_split(&mut self, split_ratio: f32) -> TrainTestSubsetData;
}

//                           x_train    y_train        # train  x_test     y_test     # test
pub type TrainTestTuple = (Array2<f64>, Array2<f64>, usize, Array2<f64>, Array2<f64>, usize);
pub struct TrainTestSubsetData(TrainTestTuple);

pub struct TrainSubsetRef<'a> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array2<f64>,
    pub size: usize,
}

pub struct TestSubsetRef<'a> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array2<f64>,
    pub size: usize,
}

pub type TrainTestSubsetRef<'a> = (TrainSubsetRef<'a>, TestSubsetRef<'a>);

impl TrainTestSubsetData {
    pub fn get_ref<'a>(&'a self) -> TrainTestSubsetRef<'a> {
        (
            TrainSubsetRef {
                x: &self.0.0,
                y: &self.0.1,
                size: self.0.2
            },
            TestSubsetRef {
                x: &self.0.3,
                y: &self.0.4,
                size: self.0.5,
            }
        )
    }
}

impl fmt::Display for TrainTestSubsetData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x_train shape {:?}, y_train shape  {:?}, x_test shape {:?}, y_test shape {:?}",
                 &self.0.0.shape(), &self.0.1.shape(), &self.0.3.shape(), &self.0.4.shape())
    }
}





