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

