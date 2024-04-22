use crate::activation::Decider;

#[derive(Clone, Debug)]
pub struct Sigmoid;

impl Decider for Sigmoid {
    fn decide(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn gradient(z: f64) -> f64 {
        let s = Self::decide(z); // <Sigmoid as Function>::decide(z);
        s * (1.0 - s)
    }
}
