use std::{fmt, str::FromStr};
use crate::optimizer::Optim;

// Types not exclusive to any module

#[derive(Debug, Copy, Clone)]
pub enum Eval { Train, Test }

#[derive(Debug, Copy, Clone)]
pub enum Batch { SGD, Mini(usize), Mini_(usize, Optim) }

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
}

impl Default for Batch {
    fn default() -> Self { Batch::SGD }
}

impl fmt::Display for Batch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Batch::SGD => write!(f, "(SGD)"),
            Batch::Mini(_) | Batch::Mini_(_, Optim::Default) => write!(f, "(MiniBatch)"),
            Batch::Mini_(_, optt) => write!(f, "(MiniBatch + {})", &optt),
        }
    }
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
