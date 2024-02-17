pub mod functions;

use std::fmt::Debug;

pub trait Activation<T> : ActivationClone<T> + Debug  {
    fn compute(&self, x: T) -> T;
    fn derivative(&self, x: T) -> T;
}

pub trait ActivationClone<T> {
    fn clone_box(&self) -> Box<dyn Activation<T>>;
}

impl<Y, X> ActivationClone<X> for Y
where
    Y: 'static + Activation<X> + Clone,
{
    fn clone_box(&self) -> Box<dyn Activation<X>> {
        Box::new(self.clone())
    }
}

// Manual implementation which defers to blanket implementation of clone_box
impl<T> Clone for Box<dyn Activation<T>> {
    fn clone(&self) -> Box<dyn Activation<T>> {
        self.clone_box()
    }
}
