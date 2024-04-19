use crate::activation::Decider;

#[derive(Clone, Debug)]
pub struct Relu;

impl Decider for Relu {
    fn decide(x: f64) -> f64 {
        x.max(0.0)
    }

    fn gradient(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
