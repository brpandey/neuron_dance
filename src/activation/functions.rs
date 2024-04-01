// To add new activation type update 3 places:
// Add new activation module type here - Start (1)
pub mod relu;
pub mod sigmoid;
pub mod tanh;
// End Finish (1)

use std::{convert::From, str::FromStr};
use crate::activation::{Activation, functions::
                        {relu::Relu, sigmoid::Sigmoid, tanh::Tanh}}; // Update (2)

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
    Relu,
    Sigmoid,
    Tanh,
}

impl From<Act> for Box<dyn Activation> {
    fn from(activation_type: Act) -> Self {
        // Add new activation type here - Start (3)
        match activation_type {
            Act::Relu => Box::new(Relu),
            Act::Sigmoid => Box::new(Sigmoid),
            Act::Tanh => Box::new(Tanh),
        }
        // End Finish (3)
    }
}
