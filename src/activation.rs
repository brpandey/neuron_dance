/// Groups the common library of simple units or activation functions
/// which when used in totality help model more complex target functions

/// Defines activation trait object interface allowing us
/// to collect the vectorized activate and derivative functions
/// of each activation type

/// The simpler activation units (functions) use parameters
/// which allow us to use gradient descent optimization to tweak
/// these values once its clear how they all work together in different
/// layers

/// The simple units make micro decisions given their params and forward
/// their answer in their output


// To add new activation type update 3 places:
// Add new activation module type here - Start (1)
pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod softmax;
pub mod leaky_relu;

use std::{convert::From, str::FromStr, fmt::Debug};
use ndarray::Array2;
use crate::activation::{relu::Relu, sigmoid::Sigmoid, tanh::Tanh,
                         softmax::Softmax, leaky_relu::LeakyRelu}; // Update (2)

use crate::weight::Weit;

pub type ActFp = fn(&Array2<f64>) -> Array2<f64>; // function pointer

pub trait Activation : Debug  {
    fn pair(&self) -> (ActFp, ActFp);
}

// blanket implementation for all function types T
impl<T: Decider + Debug + Clone + 'static> Activation for T {
    // return both activate, activate derivative func in a tuple
    fn pair(&self) -> (ActFp, ActFp) {
        (<T as Decider>::activate, <T as Decider>::derivative)
    }
}

pub trait Decider {

    // perform non-linear activation
    fn activate(z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|v| Self::decide(v))
    }

    fn decide(x: f64) -> f64 { x }
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
    Relu_(Weit), // with supplied weight initialization type
    Sigmoid,
    Sigmoid_(Weit),
    Tanh,
    Tanh_(Weit),
    Softmax,
    Softmax_(Weit),
    LeakyRelu,
}

impl From<Act> for Box<dyn Activation> {
    fn from(activation_type: Act) -> Self {
        // Add new activation type here - Start (3)
        match activation_type {
            Act::Relu | Act::Relu_(_) => Box::new(Relu),
            Act::Sigmoid | Act::Sigmoid_(_) => Box::new(Sigmoid),
            Act::Tanh | Act::Tanh_(_) => Box::new(Tanh),
            Act::Softmax | Act::Softmax_(_) => Box::new(Softmax),
            Act::LeakyRelu => Box::new(LeakyRelu),
        }
    }
}
