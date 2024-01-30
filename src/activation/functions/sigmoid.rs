use crate::activation::Activation;

#[derive(Clone, Debug)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn compute(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f64) -> f64 {
        let s = self.compute(x);
        s * (1.0 - s)
    }
}

