pub mod functions;

use std::fmt::Debug;
use ndarray::{Array2, ArrayView2};

use crate::cost::functions::FunctionCost;
use crate::activation::functions::Act;

pub type CostFp = fn(&Array2<f64>, &Array2<f64>) -> f64; // function pointer
pub type CostDFp = fn(&Array2<f64>, &ArrayView2<f64>) -> Array2<f64>; // function pointer
pub type CostCDFp = fn(Array2<f64>, Array2<f64>, Act) -> Array2<f64>; // function pointer

pub trait Cost : Debug  {
    fn triple(&self) -> (CostFp, CostDFp, CostCDFp);
}

// blanket implementation for all function types T
impl<T: FunctionCost + Debug + Clone + 'static> Cost for T {
    // return both activate, activate derivative func in a tuple
    fn triple(&self) -> (CostFp, CostDFp, CostCDFp) {
        (<T as FunctionCost>::compute, <T as FunctionCost>::derivative, <T as FunctionCost>::combine_derivative)
    }
}


