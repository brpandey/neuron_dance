use std::fmt::Debug;

pub trait Activation : ActivationClone + Debug  {
    fn apply(&self, x: f64) -> f64;
    fn apply_derivative(&self, x: f64) -> f64;
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


#[derive(Clone, Debug)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn apply_derivative(&self, x: f64) -> f64 {
        let s = self.apply(x);
        s * (1.0 - s)
    }
}

#[derive(Clone, Debug)]
pub struct Relu;

impl Activation for Relu {
    fn apply(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn apply_derivative(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
