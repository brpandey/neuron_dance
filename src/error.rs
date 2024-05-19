use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Unable to read file -- {0}")]
    FileRead(#[from] std::io::Error),
    #[error("Unable to read csv file -- {0}")]
    CSVRead(#[from] csv::Error),
    #[error("Unable to deserialize csv into ndarray -- {0}")]
    CSVBuilder(#[from] ndarray_csv::ReadError),
}
