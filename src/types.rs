use std::{fmt, str::FromStr};

// Types not exclusive to any module

#[derive(Debug, Copy, Clone)]
pub enum Eval { Train, Test }

#[derive(Debug, Copy, Clone)]
pub enum Batch { SGD, Mini(usize) }

impl Batch {
    pub fn value(&self) -> usize {
        match self {
            Batch::SGD => 1,
            Batch::Mini(ref size) => *size,
        }
    }
}

impl fmt::Display for Batch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Batch::SGD => write!(f, "(SGD)"),
            Batch::Mini(_) => write!(f, "(MiniBatch)"),
        }
    }
}


#[derive(Debug)]
pub enum Classification {
    Binary,
    MultiClass,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Mett { // Metrics Type
    Accuracy,
    Cost,
}

pub struct Metr<'a>(pub &'a str); // str format metric type specified in layer

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

#[derive(Debug, PartialEq, Eq)]
pub struct MetrParseError;

impl FromStr for Mett {
    type Err = MetrParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "accuracy" => Ok(Mett::Accuracy),
            "cost" => Ok(Mett::Cost),
            _ => Err(MetrParseError),
        }
    }
}
