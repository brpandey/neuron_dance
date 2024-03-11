use ndarray::Array2;
pub const DATASET_DIR: &str = "./src/dataset/raw/";

pub mod csv;
pub mod idx;

pub trait DataSet  {
    fn train_test_split(&self, split_ratio: f32) -> TrainTestSplitData;
}

//                           x_train    y_train        # train  x_test        y_test     # test
pub struct TrainTestSplitData(Array2<f64>, Array2<f64>, usize, Array2<f64>, Array2<f64>, usize);

pub struct TrainSplitRef<'a> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array2<f64>,
    pub size: usize,
}

pub struct TestSplitRef<'a> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array2<f64>,
    pub size: usize,
}

pub type TrainTestSplitRef<'a> = (TrainSplitRef<'a>, TestSplitRef<'a>);

impl TrainTestSplitData {
    pub fn get_ref<'a>(&'a self) -> TrainTestSplitRef<'a> {
        (
            TrainSplitRef {
                x: &self.0,
                y: &self.1,
                size: self.2
            },
            TestSplitRef {
                x: &self.3,
                y: &self.4,
                size: self.5,
            }
        )
    }
}
