pub mod functions;

use std::fmt::Debug;
use ndarray::{Array2, ArrayView2};

use crate::cost::functions::FunctionCost;

pub type CostFp = fn(&Array2<f64>, &Array2<f64>) -> f64; // function pointer
pub type CostDFp = fn(&Array2<f64>, &ArrayView2<f64>) -> Array2<f64>; // function pointer

pub trait Cost : Debug  {
    fn pair(&self) -> (CostFp, CostDFp);
}

// blanket implementation for all function types T
impl<T: FunctionCost + Debug + Clone + 'static> Cost for T {
    // return both activate, activate derivative func in a tuple
    fn pair(&self) -> (CostFp, CostDFp) {
        (<T as FunctionCost>::compute, <T as FunctionCost>::derivative)
    }
}


