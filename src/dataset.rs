use ndarray::{Array2, Axis};
use std::{env, fmt, path::Path};

use crate::algebra::AlgebraExt;
use crate::types::{Eval, SimpleError};

pub mod csv;
pub mod idx;

#[derive(Copy, Clone)]
pub enum DataSetFormat {
    CSV,
    IDX,
}

pub const ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");

pub trait DataSet {
    fn fetch(&mut self) -> Result<(), SimpleError>;
    fn head(&self);
    fn shuffle(&mut self);
    fn train_test_split(&mut self, split_ratio: f32) -> TrainTestSubsets;
}

pub fn sanitize_token(token: &str) -> Result<&str, SimpleError> {
    // ensure token doesn't have parent directory traversal in string
    if let Some(t) = Path::new(token)
        .file_name()
        .as_ref()
        .and_then(|os| os.to_str())
    {
        match t.split_once('.') {
            None => Ok(t),
            Some((a, _b)) => Ok(a),
        }
    } else {
        Err(SimpleError::PathToken(format!(
            "Path token is not well-formed {}",
            &token
        )))
    }
}

//                           x_train    y_train        # train  x_test     y_test    # test # n_features
pub type TrainTestTuple = (
    Array2<f64>,
    Array2<f64>,
    usize,
    Array2<f64>,
    Array2<f64>,
    usize,
    usize,
);

pub struct TrainTestSubsets {
    train: Subset,
    test: Subset,
    n_features: usize,
}

impl TrainTestSubsets {
    pub fn new(
        format: DataSetFormat,
        data: TrainTestTuple,
        class_names: Option<Vec<String>>,
    ) -> Self {
        let train = Subset {
            x: data.0,
            y: data.1,
            size: data.2,
            format,
            class_names: class_names.clone(),
        };

        let test = Subset {
            x: data.3,
            y: data.4,
            size: data.5,
            format,
            class_names,
        };

        Self {
            train,
            test,
            n_features: data.6,
        }
    }

    pub fn num_features(&self) -> usize {
        self.n_features
    }

    // scale is 0 min 1 max
    pub fn min_max_scale(&self, min: f64, max: f64) -> Self {
        let x_train = &self.train.x;
        let x_test = &self.test.x;

        // get the min and max values for each column
        let (x_train_mins, x_test_mins) = (x_train.min_axis(Axis(0)), x_test.min_axis(Axis(0)));
        let (x_train_maxs, x_test_maxs) = (x_train.max_axis(Axis(0)), x_test.max_axis(Axis(0)));

        let x_train_std = (x_train - &x_train_mins) / (&x_train_maxs - &x_train_mins);
        let x_test_std = (x_test - &x_test_mins) / (&x_test_maxs - &x_test_mins);

        let x_train_scaled = x_train_std * (max - min) + min;
        let x_test_scaled = x_test_std * (max - min) + min;

        let train = Subset {
            x: x_train_scaled,
            y: self.train.y.clone(),
            size: self.train.size,
            format: self.train.format,
            class_names: self.train.class_names.clone(),
        };

        let test = Subset {
            x: x_test_scaled,
            y: self.test.y.clone(),
            size: self.test.size,
            format: self.test.format,
            class_names: self.test.class_names.clone(),
        };

        Self {
            train,
            test,
            n_features: self.n_features,
        }
    }

    pub fn train(&self) -> &'_ Subset {
        &self.train
    }

    pub fn subset_ref(&self, eval: &Eval) -> &'_ Subset {
        match *eval {
            Eval::Train => &self.train,
            Eval::Test => &self.test,
        }
    }
}

impl fmt::Display for TrainTestSubsets {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "x_train shape {:?}, y_train shape  {:?}, x_test shape {:?}, y_test shape {:?}",
            &self.train.x.shape(),
            &self.train.y.shape(),
            &self.test.x.shape(),
            &self.test.y.shape()
        )
    }
}

#[derive(Clone)]
pub struct Subset {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
    pub size: usize,
    pub format: DataSetFormat,
    pub class_names: Option<Vec<String>>,
}

impl Subset {
    pub fn random(&self) -> (Array2<f64>, Array2<f64>) {
        use rand::Rng;

        let (mut rng, random_index);
        rng = rand::thread_rng();
        random_index = rng.gen_range(0..self.size);

        let x_single = self.x.select(Axis(0), &[random_index]);
        let y_single = self.y.select(Axis(0), &[random_index]);

        (x_single, y_single)
    }

    pub fn features_peek(&self, x: &Array2<f64>) {
        use crate::dataset::{csv::CSVData, idx::MnistData};
        use crate::visualize::Peek;

        match self.format {
            DataSetFormat::CSV => CSVData::peek(
                x,
                Some("=> for corresponding x input features, see tabular row"),
            ),
            DataSetFormat::IDX => {
                MnistData::peek(x, Some("=> for reduced x input image, see grid below"))
            }
        }
    }

    pub fn class_names(&self) -> Option<&Vec<String>> {
        self.class_names.as_ref()
    }
}
