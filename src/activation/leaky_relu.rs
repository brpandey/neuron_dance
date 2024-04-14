use crate::activation::FunctionAct;

#[derive(Clone, Debug)]
pub struct LeakyRelu;

impl FunctionAct for LeakyRelu {
    fn compute(x: f64) -> f64 {
        x.max(0.01*x)
    }

    fn gradient(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.01 }
    }
}
