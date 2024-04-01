use crate::activation::functions::FunctionAct;

#[derive(Clone, Debug)]
pub struct Tanh;

impl FunctionAct for Tanh {
    fn compute(x: f64) -> f64 {
        let numerator = (x).exp() - (-x).exp();
        let denometer = (x).exp() + (-x).exp();

        let res = numerator / denometer;
//        println!("tanh compute - t1 {:?}, t2 {:?}, res {:?}, x {:?}", &term1, &term2, &res, x);
        res
    }

    fn derivative(x: f64) -> f64 {
        let t = Self::compute(x); // <Tanh as Function>::compute(x);

        let res = 1.0 - (t*t);
//        println!("tanh derivative - t {:?}, res {:?}, x {:?}", &t, &res, x);
        res
    }
}
