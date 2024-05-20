use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Unable to perform IO -- {0}")]
    IO(#[from] std::io::Error),
    #[error("Unable to read csv file -- {0}")]
    CSV(#[from] csv::Error),
    #[error("Unable to deserialize csv into ndarray -- {0}")]
    CSVBuilder(#[from] ndarray_csv::ReadError),
    #[error("Conversion shape error -- {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Deserialize error -- {0}")]
    Deserialize(#[from] nanoserde::DeBinErr),
    #[error("Incorrect model operation")]
    InvalidModel,
}

impl DatasetError {
    pub fn print_and_exit(self) {
        println!("{}", &self);
        std::process::exit(1)
    }
}
