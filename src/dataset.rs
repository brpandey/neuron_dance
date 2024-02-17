use csv::ReaderBuilder as CSV;
use ndarray::{prelude::*, Axis, ScalarOperand};
use ndarray_csv::Array2Reader;
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use rand_isaac::isaac64::Isaac64Rng;
use num::Float;
use serde;

pub struct DataSet<T>(Array2<T>);

impl <T: Float + SampleUniform + ScalarOperand + for<'de> serde::de::Deserialize<'de>> DataSet<T> {
    pub fn new(path: &str) -> Self {
        let mut reader = CSV::new()
            .has_headers(true)
            .from_path(path)
            .expect("expect 1");

        let data_array: Array2<T> = reader
            .deserialize_array2_dynamic()
            .expect("can deserialize array");

        DataSet(data_array)
    }

    pub fn train_test_split(&self, split_ratio: f32) -> TrainTestSplitData<T> {

        let data = &self.0;
        let n_size = data.shape()[0]; // 1345
        let n_features = data.shape()[1]; // 4 = 3 input features + 1 outcome / target

        let seed = 42; // for reproducibility
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // take random shuffling following a normal distribution
        let shuffled = data.sample_axis_using(Axis(0), n_size as usize, SamplingStrategy::WithoutReplacement, &mut rng).to_owned();

        let n1 = (n_size as f32 * split_ratio).ceil() as usize;
        let n2 = n_size - n1;

        let mut first_raw_vec = shuffled.into_raw_vec();

        // hence the first_raw_vec is now size n1 * n_features, leaving second_raw_vec with remainder
        let second_raw_vec = first_raw_vec.split_off(n1 * n_features); 

        let train_data = Array2::from_shape_vec((n1, n_features), first_raw_vec).unwrap();
        let test_data = Array2::from_shape_vec((n2, n_features), second_raw_vec).unwrap();

        let (x_train, y_train) = (
            train_data.slice(s![.., 0..3]).to_owned() / T::from(255.0).unwrap(),
            train_data.slice(s![.., 3..4]).to_owned(),
            //        train_data.column(3).to_owned(),
        );

        let (x_test, y_test) : (Array2<T>, Array2<T>) = (
            test_data.slice(s![.., 0..3]).to_owned() / T::from(255.0).unwrap(),
            test_data.slice(s![.., 3..4]).to_owned(),
            //        test_data.column(3).to_owned(),
        );

        TrainTestSplitData(x_train, y_train, n1, x_test, y_test, n2)
    }
}

//                           x_train    y_train        # train  x_test        y_test     # test
pub struct TrainTestSplitData<T>(Array2<T>, Array2<T>, usize, Array2<T>, Array2<T>, usize);

pub struct TrainSplitRef<'a, T> {
    pub x: &'a Array2<T>,
    pub y: &'a Array2<T>,
    pub size: usize,
}

pub struct TestSplitRef<'a, T> {
    pub x: &'a Array2<T>,
    pub y: &'a Array2<T>,
    pub size: usize,
}

pub type TrainTestSplitRef<'a, T> = (TrainSplitRef<'a, T>, TestSplitRef<'a, T>);

impl <T> TrainTestSplitData<T> {
    pub fn get_ref<'a>(&'a self) -> TrainTestSplitRef<'a, T> {
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
