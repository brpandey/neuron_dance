// To add new activation type update 3 places:
// Add new activation module type here - Start (1)
pub mod relu;
pub mod sigmoid;
// End Finish (1)

use std::str::FromStr;
use std::fmt;
use crate::activation::{Activation,
                        functions::{relu::Relu, sigmoid::Sigmoid}}; // Update (2)

#[derive(Debug, PartialEq, Eq)]
pub struct ActivationParseError;

// manual error msg implementation rather than pulling in external crate
impl fmt::Display for ActivationParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unknown activation type")
    }
}

impl<T: std::fmt::Debug + num::Float + 'static > FromStr for Box<dyn Activation<T>> {
    type Err = ActivationParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Add new activation type here - Start (3)
        match s {
            "relu" => Ok(Box::new(Relu)),
            "sigmoid" => Ok(Box::new(Sigmoid)),
            _ => Err(ActivationParseError),
        }
        // End Finish (3)
    }
}
