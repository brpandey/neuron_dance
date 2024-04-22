use crate::activation::Decider;

#[derive(Clone, Debug)]
pub struct Relu;

impl Decider for Relu {
    fn decide(z: f64) -> f64 {
        z.max(0.0)
    }

    fn gradient(z: f64) -> f64 {
        if z > 0.0 { 1.0 } else { 0.0 }
    }
}
