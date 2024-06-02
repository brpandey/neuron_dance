use crate::activation::Decider;

#[derive(Clone, Debug)]
pub struct LeakyRelu;

impl Decider for LeakyRelu {
    fn decide(z: f64) -> f64 {
        z.max(0.01 * z)
    }

    fn gradient(z: f64) -> f64 {
        if z > 0.0 {
            1.0
        } else {
            0.01
        }
    }
}
