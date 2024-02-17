use std::fmt::Debug;
use num::Float;
use crate::activation::functions::Function;

#[derive(Clone, Debug)]
pub struct Relu; // pass through struct doesn't hold any state

impl <T: Float + Debug + 'static> Function<T> for Relu {
    fn compute(x: T) -> T {
        x.max(T::from(0.0).unwrap())
    }

    fn derivative(x: T) -> T {
        let zero = T::from(0.0).unwrap();
        let one = T::from(1.0).unwrap();
        if x > zero { one } else { zero }
    }
}
