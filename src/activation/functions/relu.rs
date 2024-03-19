use crate::activation::functions::FunctionAct;

#[derive(Clone, Debug)]
pub struct Relu;

impl FunctionAct for Relu {
    fn compute(x: f64) -> f64 {
        x.max(0.0)
    }

    fn derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
