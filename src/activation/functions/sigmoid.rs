use crate::activation::functions::FunctionAct;

#[derive(Clone, Debug)]
pub struct Sigmoid;

impl FunctionAct for Sigmoid {
    fn compute(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(x: f64) -> f64 {
        let s = Self::compute(x); // <Sigmoid as Function>::compute(x);
        s * (1.0 - s)
    }
}
