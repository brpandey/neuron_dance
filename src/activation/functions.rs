// To add new activation type update 3 places:
// Add new activation module type here - Start (1)
pub mod relu;
pub mod sigmoid;
pub mod tanh;
// End Finish (1)

use std::{convert::From, str::FromStr};
use crate::activation::{Activation, functions::
                        {relu::Relu, sigmoid::Sigmoid, tanh::Tanh}}; // Update (2)

use crate::weight::Weit;

pub trait FunctionAct {
    fn compute(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
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
            _ => Err(ActivationParseError),
        }
        // End Finish (3)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Act {
    Relu, // uses default weight initialization type
    Relu_(Weit), // with supplied weight initialization type
    Sigmoid,
    Sigmoid_(Weit),
    Tanh,
    Tanh_(Weit),
}

impl From<Act> for Box<dyn Activation> {
    fn from(activation_type: Act) -> Self {
        // Add new activation type here - Start (3)
        match activation_type {
            Act::Relu | Act::Relu_(_) => Box::new(Relu),
            Act::Sigmoid | Act::Sigmoid_(_) => Box::new(Sigmoid),
            Act::Tanh | Act::Tanh_(_) => Box::new(Tanh),
        }
        // End Finish (3)
    }
}
