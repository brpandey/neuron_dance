pub mod functions;

use std::fmt::Debug;
use crate::activation::functions::Function;

pub type MathFp = fn(f64) -> f64; // function pointer

pub trait Activation : Debug  {
    fn pair(&self) -> (MathFp, MathFp);
}

// blanket implementation for all function types T
impl<T: Function + Debug + Clone + 'static> Activation for T {
    // return both activate, activate derivative func in a tuple
    fn pair(&self) -> (MathFp, MathFp) {
        (<T as Function>::compute, <T as Function>::derivative)
    }
}

#[derive(Copy, Clone)]
pub enum Act {
    Relu,
    Sigmoid,
}
