/// Optimizer
/// Optimizers help to facilitate a faster gradient descent optimization
/// by utilizing more contextual information to speed up the gradient descent process

/// Gradient descent optimization:
/// θ = θ − η ⋅ ∇θ T(θ).
/// Minimize target function T by updating parameters to opposite direction of
/// derivative of the target function with respect to the params (e.g w, b)
/// Hence w - η ⋅ dT/dw or b - η ⋅ dT/db)

pub mod adam;
pub mod amsgrad;
pub mod nadam;

use std::borrow::Cow;
use ndarray::Array2;

use crate::optimizer::{adam::Adam, amsgrad::AmsGrad, nadam::NAdam};

#[derive(Debug, Copy, Clone)]
pub enum Optim {
    Default,
    Adam,
    AMSGrad,
    NAdam,
}

#[derive(Eq, Hash, PartialEq, Debug, Copy, Clone)]
pub enum ParamKey {
    WeightGradient(u8),
    BiasGradient(u8),
}

#[derive(Eq, Hash, PartialEq, Debug, Copy, Clone)]
pub enum HistType { // Historical Type
    Mean,
    Variance,
    Vhat,
}

#[derive(Eq, Hash, PartialEq, Debug, Copy, Clone)]
pub struct CompositeKey(ParamKey, HistType);

impl std::fmt::Display for Optim { // use debug fmt imp for display
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait Optimizer {
    fn calculate<'a>(&mut self, _key: ParamKey, value: &'a Array2<f64>, _t: usize) -> Cow<'a, Array2<f64>> {
        Cow::Borrowed(value)
    }
}

impl From<Optim> for Box<dyn Optimizer> {
    fn from(optimzer_type: Optim) -> Self {
        match optimzer_type {
            Optim::Default => Box::new(Default),
            Optim::Adam => Box::new(Adam::new()),
            Optim::AMSGrad => Box::new(AmsGrad::new()),
            Optim::NAdam => Box::new(NAdam::new()),
        }
    }
}

impl std::fmt::Debug for Box<dyn Optimizer> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Some optimizer variant")
    }
}

#[derive(Debug)]
pub struct Default; // default optimizer is none

impl Optimizer for Default {}
