pub mod functions;

use std::fmt::Debug;
use crate::activation::functions::FunctionAct;

pub type ActFp = fn(f64) -> f64; // function pointer

pub trait Activation : Debug  {
    fn pair(&self) -> (ActFp, ActFp);
}

// blanket implementation for all function types T
impl<T: FunctionAct + Debug + Clone + 'static> Activation for T {
    // return both activate, activate derivative func in a tuple
    fn pair(&self) -> (ActFp, ActFp) {
        (<T as FunctionAct>::compute, <T as FunctionAct>::derivative)
    }
}

