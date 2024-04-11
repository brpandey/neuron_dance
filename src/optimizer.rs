use std::borrow::Cow;
use ndarray::Array2;

pub mod adam;

use crate::optimizer::adam::Adam;

#[derive(Debug, Copy, Clone)]
pub enum Optim {
    Default,
    Adam,
}

impl std::fmt::Display for Optim { // use debug fmt imp for display
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait Optimizer {
    fn calculate<'a>(&mut self, _key: String, value: &'a Array2<f64>, _t: usize) -> Cow<'a, Array2<f64>> {
        Cow::Borrowed(value)
    }
}

impl From<Optim> for Box<dyn Optimizer> {
    fn from(optimzer_type: Optim) -> Self {
        match optimzer_type {
            Optim::Default => Box::new(Default),
            Optim::Adam => Box::new(Adam::new()),
        }
    }
}

impl std::fmt::Debug for Box<dyn Optimizer> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Some optimizer variant")
    }
}


pub struct Default; // default optimizer is none

impl Optimizer for Default {}
