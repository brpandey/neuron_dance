use csv::ReaderBuilder as CSV;

use ndarray::{prelude::*, Axis};
use ndarray_csv::Array2Reader;

use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::isaac64::Isaac64Rng;

pub struct DataSet(Array2<f64>);

impl DataSet {
    pub fn new(path: &str) -> Self {
        let mut reader = CSV::new()
            .has_headers(true)
            .from_path(path)
            .expect("expect 1");

        let data_array: Array2<f64> = reader
            .deserialize_array2_dynamic()
            .expect("can deserialize array");

        println!("array is {:?}, shape is {:?}", &data_array, data_array.shape());

        DataSet(data_array)
    }

    pub fn train_test_split(&self, split_ratio: f32) -> ((Array2<f64>, Array2<f64>), (Array2<f64>, Array2<f64>), (usize, usize)) {

        let data = &self.0;
        let n_size = data.shape()[0]; // 1345
        let n_features = data.shape()[1]; // 4 = 3 input features + 1 outcome / target

        let seed = 42; // for reproducibility
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // take random shuffling following a normal distribution
        let shuffled = data.sample_axis_using(Axis(0), n_size as usize, SamplingStrategy::WithoutReplacement, &mut rng).to_owned();
        println!("shuffled {:?}", &shuffled);

        // end shuffle array

        let n1 = (n_size as f32 * split_ratio).ceil() as usize;
        let n2 = n_size - n1;

        let mut first_raw_vec = shuffled.into_raw_vec();

        // hence the first_raw_vec is now size n1 * n_features, leaving second_raw_vec with remainder
        let second_raw_vec = first_raw_vec.split_off(n1 * n_features); 

        let train_data = Array2::from_shape_vec((n1, n_features), first_raw_vec).unwrap();
        let test_data = Array2::from_shape_vec((n2, n_features), second_raw_vec).unwrap();

        let (x_train, y_train) = (
            train_data.slice(s![.., 0..3]).to_owned() / 255.0,
            train_data.slice(s![.., 3..4]).to_owned(),
            //        train_data.column(3).to_owned(),
        );

        let (x_test, y_test) : (Array2<f64>, Array2<f64>) = (
            test_data.slice(s![.., 0..3]).to_owned() / 255.0,
            test_data.slice(s![.., 3..4]).to_owned(),
            //        test_data.column(3).to_owned(),
        );

        ((x_train, y_train), (x_test, y_test), (n1, n2))
    }
}
