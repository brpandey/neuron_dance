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

pub mod functions;

use std::fmt::Debug;
use ndarray::Array2;

use crate::activation::functions::FunctionAct;

pub type ActFp = fn(&Array2<f64>) -> Array2<f64>; // function pointer

pub trait Activation : Debug  {
    fn pair(&self) -> (ActFp, ActFp);
}

// blanket implementation for all function types T
impl<T: FunctionAct + Debug + Clone + 'static> Activation for T {
    // return both activate, activate derivative func in a tuple
    fn pair(&self) -> (ActFp, ActFp) {
        (<T as FunctionAct>::activate, <T as FunctionAct>::derivative)
    }
}

