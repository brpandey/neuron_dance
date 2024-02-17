use num::Float;
use std::fmt::Debug;
use crate::activation::functions::Function;

#[derive(Clone, Debug)]
pub struct Sigmoid;

impl <T: Float + Debug + 'static> Function<T> for Sigmoid {
    fn compute(x: T) -> T {
        let one = T::from(1.0).unwrap();
        one / (one + (-x).exp())
    }

    fn derivative(x: T) -> T {
        let one = T::from(1.0).unwrap();
        let s = Self::compute(x); // <Sigmoid as Function>::compute(x);
        s * (one - s)
    }
}
