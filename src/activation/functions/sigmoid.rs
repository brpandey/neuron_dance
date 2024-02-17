use num::Float;
use std::fmt::Debug;
//use std::marker::PhantomData;
use crate::activation::Activation;

#[derive(Clone, Debug)]
pub struct Sigmoid;
// pub struct Sigmoid<T>(PhantomData<T>);

/*
impl<T> Default for Sigmoid<T> {
    fn default() -> Self {
        Sigmoid(PhantomData)
    }
}
*/

impl <T: Float + Debug + 'static> Activation<T> for Sigmoid {
    fn compute(&self, x: T) -> T {
        let one = T::from(1.0).unwrap();
        one / (one + (-x).exp())
    }

    fn derivative(&self, x: T) -> T {
        let one = T::from(1.0).unwrap();
        let s = self.compute(x);
        s * (one - s)
    }
}
