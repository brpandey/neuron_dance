use csv::ReaderBuilder as Builder;
use ndarray::{Array2, Axis, s, Data, Ix2, ArrayBase};
use ndarray_csv::Array2Reader;
use ndarray_rand::{RandomExt, rand::SeedableRng};
use rand_isaac::isaac64::Isaac64Rng;
use ndarray_rand::SamplingStrategy::WithoutReplacement as strategy;

use crate::dataset::{sanitize_token, ROOT_DIR, DataSet, DataSetFormat, TrainTestSubsets, TrainTestTuple};
use crate::visualize::{Visualize, Peek, Empty};
use crate::types::SimpleError;

#[derive(Default)]
pub struct CSVData<'a> {
    ctype: CSVType<'a>,
    scale: f64,
    class_names: Option<Vec<String>>,
    data: Option<Array2<f64>>,
    headers: Option<Vec<String>>,
}

#[derive(Default)]
pub enum CSVType<'a> {
    #[default]
    RGB,
    Custom(&'a str, Option<Vec<&'a str>>), // filename w/o csv suffix and class names if relevant
}

impl <'a> CSVType<'a> {
    pub fn filename(&self) -> Result<String, SimpleError> {
        let token = match self {
            CSVType::RGB => "rgb",
            CSVType::Custom(t, _) => sanitize_token(t)?,
        };

        Ok(format!("{}/data/csv/{}.csv", ROOT_DIR, token))
    }
}

impl <'b> CSVData<'b> {
    pub fn new<'a>(ctype: CSVType<'a>) -> Self
    where 'a: 'b {
        match ctype {
            CSVType::RGB => Self { ctype, scale: 256.0, ..Default::default()},
            CSVType::Custom(_, ref names) => {
                let cn = names.as_ref()
                    .map(|v| v.iter().map(|s| s.to_string()).collect());
                Self { ctype, scale: 1.0, class_names: cn, ..Default::default() }
            },
        }
    }
}

impl Peek for CSVData<'_> {
    fn peek<S: Data<Elem = f64>>(x: &ArrayBase<S, Ix2>, text: Option<&str>) {
        Visualize::table_preview(x, None::<Empty>, false, text);
    }
}

impl <'b> DataSet for CSVData<'b> {

    fn fetch(&mut self) -> Result<(), SimpleError> {
        if self.data.is_some() { return Ok(()) }; // fetch unless already previously fetched data

        let token = &self.ctype.filename()?;

        let mut reader = Builder::new()
            .has_headers(true)
            .from_path(token)?;

        let headers: Vec<String> = reader.headers().unwrap().iter()
            .map(|s| s.to_owned()).collect();

        let data_array: Array2<f64> = reader.deserialize_array2_dynamic()?;

        self.data = Some(data_array);
        self.headers = Some(headers);

        Ok(())
    }

    fn head(&self) {
        if self.data.is_none() { return } // ensure data has been fetched before performing head preview
        Visualize::table_preview(
            self.data.as_ref().unwrap(),
            self.headers.as_deref(), false, Some("> head csv-file")
        );
    }

    fn shuffle(&mut self) {
        if self.data.is_none() { return } // ensure data has been fetched before shuffling

        let data = self.data.as_mut().unwrap();
        let seed = 42; // for reproducibility
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let n_size = data.shape()[0];

        // take random shuffling following a normal distribution
        let shuffled = data.sample_axis_using(Axis(0), n_size, strategy, &mut rng).to_owned();
        self.data = Some(shuffled);
    }

    fn train_test_split(&mut self, split_ratio: f32) -> TrainTestSubsets {
        if self.data.is_none() {
            let _ = self.fetch();
        }

        let scale = self.scale;
        let data = self.data.as_mut().unwrap();
        let class_names = self.class_names.clone();

        let n_size = data.shape()[0]; // # rows
        let n_columns = data.shape()[1]; //# columns (inc target)
        let n_features = n_columns-1;

        let n1 = (n_size as f32 * split_ratio).ceil() as usize;
        let n2 = n_size - n1;

        let mut first_raw_vec = data.clone().into_raw_vec();

        // hence the first_raw_vec is now size n1 * n_features, leaving second_raw_vec with remainder
        let second_raw_vec = first_raw_vec.split_off(n1 * n_columns); 

        let train_data = Array2::from_shape_vec((n1, n_columns), first_raw_vec).unwrap();
        let test_data = Array2::from_shape_vec((n2, n_columns), second_raw_vec).unwrap();

        let (s1, e1) = (0, n_features); // train data
        let e2 = n_columns; // label data (last data column)

        let (x_train, y_train) = (
            train_data.slice(s![.., s1..e1]).to_owned() / scale,
            train_data.slice(s![.., e1..e2]).to_owned(),
        );

        let (x_test, y_test) : (Array2<f64>, Array2<f64>) = (
            test_data.slice(s![.., s1..e1]).to_owned() / scale,
            test_data.slice(s![.., e1..e2]).to_owned(),
        );

        let ttt: TrainTestTuple = (x_train, y_train, n1, x_test, y_test, n2, n_features);
        let tts = TrainTestSubsets::new(DataSetFormat::CSV, ttt, class_names);

        println!("Data subset shapes {}\n", &tts);

        tts
    }
}
