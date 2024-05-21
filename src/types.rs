use nanoserde::{DeBin, SerBin};
use thiserror::Error;

use crate::optimizer::Optim;
// Types not exclusive to any module

#[derive(Debug, Copy, Clone, strum::EnumIter)]
pub enum Eval { Train, Test }

#[derive(Debug, Copy, Clone, strum_macros::Display, strum_macros::EnumString, DeBin, SerBin)]
pub enum Batch {
    SGD,
    Mini(usize),
    Mini_(usize, Optim)
}

impl Batch {
    pub fn is_mini(&self) -> bool {
        if let Batch::Mini(_) | Batch::Mini_(_,_) = self { return true }
        return false
    }

    pub fn value(&self) -> usize {
        match self {
            Batch::SGD => 1,
            Batch::Mini(ref size) | Batch::Mini_(ref size, _) => *size,
        }
    }

    pub fn text_display(&self) -> String {
        match self {
            Batch::SGD => format!("(SGD)"),
            Batch::Mini(_) | Batch::Mini_(_, Optim::Default) => format!("(MiniBatch)"),
            Batch::Mini_(_, optt) => format!("(MiniBatch + {})", &optt),
        }
    }
}

impl Default for Batch {
    fn default() -> Self { Batch::SGD }
}

#[derive(Debug)]
pub enum Classification {
    Binary,
    MultiClass(usize),
}

impl Classification {
    pub fn new(size: usize) -> Self {
        if size > 1 {
            Classification::MultiClass(size)
        } else {
            Classification::Binary
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, strum_macros::Display, strum_macros::EnumString)]
pub enum Mett { // Metrics Type
    #[strum(ascii_case_insensitive)]
    Accuracy,
    #[strum(ascii_case_insensitive)]
    Cost,
}

impl Default for Mett {
    fn default() -> Self { Mett::Accuracy }
}

pub struct Metr<'a>(pub &'a str); // str format metric type specified in layer

impl<'a> Default for Metr<'a> {
    fn default() -> Self { Metr("accuracy") }
}

impl<'a> Metr<'a> {
    pub fn to_vec(&mut self) -> Vec<Mett> { // convert Metr to a collection of Mett's
        let text = self.0;
        let vec: Vec<&str> = text.split(",").map(|t| t.trim()).collect();

        vec.into_iter().fold(vec![], |mut acc, v| {
            if let Ok(m) = v.parse() {
                acc.push(m);
            }
            acc
        })
    }
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub enum ModelState { // models a linear sequence of state progression / also used to resemble operations
    #[default]
    EMPTY = 0, // start state
    ADD = 1, // valid operations => add layers or compile
    COMPILE = 2, // valid operations => fit
    FIT = 3, // valid operations => eval or predict
    EVAL = 4, // used as a comparative state, never set to this value
}

impl ModelState {
    pub fn check_valid_state(&self, op: &ModelState) -> Result<bool, SimpleError> {
        let cur = *self;

        // explicitly state allowable conditions
        match *op {
            ModelState::ADD if cur == ModelState::ADD || cur == ModelState::EMPTY => Ok(true),
            ModelState::COMPILE if cur == ModelState::ADD => Ok(true),
            ModelState::FIT if cur == ModelState::COMPILE => Ok(true),
            ModelState::EVAL if cur == ModelState::FIT => Ok(true),
            _ => {
                let txt = format!("Invalid model operation {:?} given current model state {:?}", op, self);
                Err(SimpleError::InvalidModel(txt))
            },
        }
    }
}

#[derive(Error, Debug)]
pub enum SimpleError {
    #[error("Unable to perform IO -- {0}")]
    IO(#[from] std::io::Error),
    #[error("Unable to process csv file -- {0}")]
    CSV(#[from] csv::Error),
    #[error("Unable to deserialize csv into ndarray -- {0}")]
    CSVBuilder(#[from] ndarray_csv::ReadError),
    #[error("Conversion ndarray shape error -- {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Deserialize error -- {0}")]
    Deserialize(#[from] nanoserde::DeBinErr),
    #[error("Invalid model operation -- {0}")]
    InvalidModel(String),
    #[error(transparent)]
    Unexpected(#[from] Box<dyn std::error::Error>),
}

impl SimpleError {
    pub fn print_and_exit(self) {
        println!("{}", &self);
        std::process::exit(1)
    }
}
