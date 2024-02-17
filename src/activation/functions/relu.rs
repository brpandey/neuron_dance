use std::fmt::Debug;

use num::Float;
use crate::activation::Activation;

#[derive(Clone, Debug)]
pub struct Relu;

impl <T: Float + Debug + 'static> Activation<T> for Relu {
    fn compute(&self, x: T) -> T {
        x.max(T::from(0.0).unwrap())
    }

    fn derivative(&self, x: T) -> T {
        let zero = T::from(0.0).unwrap();
        let one = T::from(1.0).unwrap();
        if x > zero { one } else { zero }
    }
}
