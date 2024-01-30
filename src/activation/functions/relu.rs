use crate::activation::Activation;

#[derive(Clone, Debug)]
pub struct Relu;

impl Activation for Relu {
    fn compute(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

