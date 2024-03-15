use csv::ReaderBuilder as Builder;
use ndarray::{Array2, Axis, s};
use ndarray_csv::Array2Reader;
use ndarray_rand::{RandomExt, SamplingStrategy, rand::SeedableRng};
use rand_isaac::isaac64::Isaac64Rng;

use crate::dataset::{DATASET_DIR, DataSet, TrainTestSubsetData};

pub type CSVBox = Box<Array2<f64>>;
pub struct CSVData(CSVType, Option<CSVBox>);

pub enum CSVType {
    RGB,
//    Custom(String),
}

impl CSVType {
    pub fn filename(&self) -> String {
        match self {
            CSVType::RGB => format!("{}{}.csv", DATASET_DIR, "rgb"),
//            CSVType::Custom => 
        }
    }
}

impl CSVData {
    pub fn new(ctype: CSVType) -> Self {
        CSVData(ctype, None)
    }
}

impl DataSet for CSVData {
    fn fetch(&mut self, token: &str) {
        let mut reader = Builder::new()
            .has_headers(true)
            .from_path(token)
            .expect("csv error");

        let data_array: Array2<f64> = reader
            .deserialize_array2_dynamic()
            .expect("can deserialize array");

        self.1 = Some(Box::new(data_array))
    }


    fn train_test_split(&mut self, split_ratio: f32) -> TrainTestSubsetData {
        if self.1.is_none() {
            self.fetch(&self.0.filename());
        }

        let data = self.1.as_mut().unwrap();

        let n_size = data.shape()[0]; // 1345
        let n_features = data.shape()[1]; // 4, => 3 input features + 1 outcome / target

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
            train_data.slice(s![.., 0..3]).to_owned() / 255.0,
            train_data.slice(s![.., 3..4]).to_owned(),
            //        train_data.column(3).to_owned(),
        );

        let (x_test, y_test) : (Array2<f64>, Array2<f64>) = (
            test_data.slice(s![.., 0..3]).to_owned() / 255.0,
            test_data.slice(s![.., 3..4]).to_owned(),
            //        test_data.column(3).to_owned(),
        );

        // For example: n_features is 4
        // x_train shape is [897, 3], y_train shape is [897, 1], x_test shape is [448, 3], y_test shape is [448, 1]
        let tts = TrainTestSubsetData((x_train, y_train, n1, x_test, y_test, n2));
        println!("Train test subset shapes are {}", &tts);
        tts
    }
}
