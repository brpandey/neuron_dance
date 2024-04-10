use std::borrow::Cow;
use ndarray::Array2;

pub mod adam;

use crate::optimizer::adam::Adam;

#[derive(Debug)]
pub enum Optt {
    Default,
    Adam,
}

pub trait Optimizer {
    fn calculate<'a>(&mut self, _key: String, value: &'a Array2<f64>, _t: usize) -> Cow<'a, Array2<f64>> {
        Cow::Borrowed(value)
    }
}

impl From<Optt> for Box<dyn Optimizer> {
    fn from(optimzer_type: Optt) -> Self {
        match optimzer_type {
            Optt::Default => Box::new(Default),
            Optt::Adam => Box::new(Adam::new()),
        }
    }
}

impl std::fmt::Debug for Box<dyn Optimizer> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Some optimizer variant")
    }
}


pub struct Default;

impl Optimizer for Default {}
