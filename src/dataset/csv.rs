use csv::ReaderBuilder as Builder;
use ndarray::{Array2, Axis, s};
use ndarray_csv::Array2Reader;
use ndarray_rand::{RandomExt, rand::SeedableRng};
use rand_isaac::isaac64::Isaac64Rng;
use ndarray_rand::SamplingStrategy::WithoutReplacement as strategy;

use crate::dataset::{ROOT_DIR, DataSet, DataSetFormat, TrainTestSubsetData, TrainTestTuple};
use crate::visualize::{Visualize, Peek};
use crate::error::DatasetError;

//                       ctype     scale    data             headers               class_names
pub struct CSVData<'a>(CSVType<'a>, f64, Option<Array2<f64>>, Option<Vec<String>>, Option<Vec<String>>);

pub enum CSVType<'a> {
    RGB,
    Custom(&'a str, Option<Vec<&'a str>>), // filename w/o csv suffix and class names if relevant
}

impl <'a> CSVType<'a> {
    pub fn filename(&self) -> String {
        let token = match self {
            CSVType::RGB => "rgb",
            CSVType::Custom(ref t, _) => CSVType::sanitize(t),
        };

        format!("{}/data/csv/{}.csv", ROOT_DIR, token)
    }

    pub fn sanitize(token: &str) -> &str {
        use std::path::Path;
        // ensure token doesn't have parent directory traversal in string
        if let Some(t) = Path::new(token).file_name().as_ref().and_then(|os| os.to_str()) {
            match t.split_once('.') {
                None => t,
                Some((a, _b)) => a
            }
        } else {
            panic!("Unable to retrieve custom csv file");
        }
    }
}

impl <'b> CSVData<'b> {
    pub fn new<'a>(ctype: CSVType<'a>) -> Self
    where 'a: 'b {
        match ctype {
            CSVType::RGB => Self(ctype, 256.0, None, None, None),
            CSVType::Custom(_, ref names) => {
                let n = names.as_ref().map(|v| v.iter().map(|s| s.to_string()).collect());
                Self(ctype, 1.0, None, None, n)
            },
        }
    }
}

impl Peek for CSVData<'_> {
    fn peek(x: &Array2<f64>, text: Option<&str>) {
        Visualize::table_preview(&x.view(), None, false, text);
    }
}

impl <'b> DataSet for CSVData<'b> {

    fn fetch(&mut self) -> Result<(), DatasetError> {
        let token = &self.0.filename();

        let mut reader = Builder::new()
            .has_headers(true)
            .from_path(token).map_err(DatasetError::CSVRead)?;

        let headers: Vec<String> = reader.headers().unwrap().iter()
            .map(|s| s.to_owned()).collect();

        let data_array: Array2<f64> = reader
            .deserialize_array2_dynamic().map_err(DatasetError::CSVBuilder)?;

        self.2 = Some(data_array);
        self.3 = Some(headers);

        Ok(())
    }

    fn head(&self) {
        if self.2.is_none() { return } // if data hasn't been fetched, return early
        Visualize::table_preview(&self.2.as_ref().unwrap().view(), self.3.as_ref(), false, Some("> head csv-file"));
    }

    fn shuffle(&mut self) {
        if self.2.is_none() { return } // if data hasn't been fetched, return early

        let data = self.2.as_mut().unwrap();
        let seed = 42; // for reproducibility
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let n_size = data.shape()[0] as usize; // 1345

        // take random shuffling following a normal distribution
        let shuffled = data.sample_axis_using(Axis(0), n_size, strategy, &mut rng).to_owned();
        self.2 = Some(shuffled);
    }

    fn train_test_split(&mut self, split_ratio: f32) -> TrainTestSubsetData {
        if self.2.is_none() {
            let _ = self.fetch();
        }

        let scale = self.1;
        let data = self.2.as_mut().unwrap();
        let headers = self.3.clone();
        let class_names = self.4.clone();

        let n_size = data.shape()[0]; // 1345
        let n_features = data.shape()[1]; // 4, => 3 input features + 1 outcome / target (columns)

        let n1 = (n_size as f32 * split_ratio).ceil() as usize;
        let n2 = n_size - n1;

        let mut first_raw_vec = data.clone().into_raw_vec();

        // hence the first_raw_vec is now size n1 * n_features, leaving second_raw_vec with remainder
        let second_raw_vec = first_raw_vec.split_off(n1 * n_features); 

        let train_data = Array2::from_shape_vec((n1, n_features), first_raw_vec).unwrap();
        let test_data = Array2::from_shape_vec((n2, n_features), second_raw_vec).unwrap();

        let (s1, e1) = (0, n_features-1); // train data
        let e2 = n_features; // label data (last data column)

        let (x_train, y_train) = (
            train_data.slice(s![.., s1..e1]).to_owned() / scale,
            train_data.slice(s![.., e1..e2]).to_owned(),
        );

        let (x_test, y_test) : (Array2<f64>, Array2<f64>) = (
            test_data.slice(s![.., s1..e1]).to_owned() / scale,
            test_data.slice(s![.., e1..e2]).to_owned(),
        );

        let ttt: TrainTestTuple = (x_train, y_train, n1, x_test, y_test, n2);
        let tts = TrainTestSubsetData { format: DataSetFormat::CSV, headers, data: ttt, class_names };

        println!("Data subset shapes {}\n", &tts);

        tts
    }
}
