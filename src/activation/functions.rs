// To add new activation type update 3 places:
// Add new activation module type here - Start (1)
pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod softmax;
pub mod leaky_relu;
// End Finish (1)

use std::{convert::From, str::FromStr};
use ndarray::Array2;
use crate::activation::{Activation, functions::
                        {relu::Relu, sigmoid::Sigmoid, tanh::Tanh,
                        softmax::Softmax, leaky_relu::LeakyRelu}}; // Update (2)

use crate::weight::Weit;

pub trait FunctionAct {

    // perform non-linear activation
    fn activate(z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| Self::compute(v))
    }

    fn compute(x: f64) -> f64 { x }
    fn gradient(x: f64) -> f64 { x }

    fn derivative(z: &Array2<f64>) -> Array2<f64> { // derivative is vectorized
        z.mapv(|v| Self::gradient(v))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ActivationParseError;

impl FromStr for Box<dyn Activation> {
    type Err = ActivationParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Add new activation type here - Start (3)
        match s {
            "relu" => Ok(Box::new(Relu)),
            "sigmoid" => Ok(Box::new(Sigmoid)),
            "tanh" => Ok(Box::new(Tanh)),
            "softmax" => Ok(Box::new(Softmax)),
            "leaky_relu" => Ok(Box::new(LeakyRelu)),
            _ => Err(ActivationParseError),
        }
        // End Finish (3)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Act {
    Relu, // uses default weight initialization type
    ReluW(Weit), // with supplied weight initialization type
    Sigmoid,
    SigmoidW(Weit),
    Tanh,
    TanhW(Weit),
    Softmax,
    LeakyRelu,
}

impl From<Act> for Box<dyn Activation> {
    fn from(activation_type: Act) -> Self {
        // Add new activation type here - Start (3)
        match activation_type {
            Act::Relu | Act::ReluW(_) => Box::new(Relu),
            Act::Sigmoid | Act::SigmoidW(_) => Box::new(Sigmoid),
            Act::Tanh | Act::TanhW(_) => Box::new(Tanh),
            Act::Softmax => Box::new(Softmax),
            Act::LeakyRelu => Box::new(LeakyRelu),
        }
        // End Finish (3)
    }
}
