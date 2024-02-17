pub mod functions;

use std::fmt::Debug;
use crate::activation::functions::Function;

pub type MathFp<T> = fn(T) -> T; // function pointer

// Activation trait
pub trait Activation<T> : Debug  {
    fn pair(&self) -> (MathFp<T>, MathFp<T>);
}

// Note:
// The Activation trait is used as a trait object in order to store each of the neural network layer
// activations in a collection.  Works in conjunction with the Function Trait
// to retrieve the necessary function pointers for relevant computations

// Note:
// The Function trait doesn't need a &self parameter and is easier to add new activation function types

// Blanket implementation for all function types U (e.g. Relu, Sigmoid)
impl<U: Function<T> + Debug + Clone + 'static, T> Activation<T> for U {
    // return both activate, activate derivative func in a tuple
    fn pair(&self) -> (MathFp<T>, MathFp<T>) {
        (<U as Function<T>>::compute, <U as Function<T>>::derivative)
    }
}
