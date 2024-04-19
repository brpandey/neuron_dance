use crate::activation::Decider;

#[derive(Clone, Debug)]
pub struct Sigmoid;

impl Decider for Sigmoid {
    fn decide(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn gradient(x: f64) -> f64 {
        let s = Self::decide(x); // <Sigmoid as Function>::decide(x);
        s * (1.0 - s)
    }
}
