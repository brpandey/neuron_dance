pub mod functions;

use std::fmt::Debug;

pub trait Activation : ActivationClone + Debug  {
    fn compute(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

pub trait ActivationClone {
    fn clone_box(&self) -> Box<dyn Activation>;
}

impl<T> ActivationClone for T
where
    T: 'static + Activation + Clone,
{
    fn clone_box(&self) -> Box<dyn Activation> {
        Box::new(self.clone())
    }
}

// Manual implementation which defers to blanket implementation of clone_box
impl Clone for Box<dyn Activation> {
    fn clone(&self) -> Box<dyn Activation> {
        self.clone_box()
    }
}
