use crate::activation::Decider;

#[derive(Clone, Debug)]
pub struct Tanh;

impl Decider for Tanh {
    fn decide(x: f64) -> f64 {
        let numerator = (x).exp() - (-x).exp();
        let denometer = (x).exp() + (-x).exp();

        let res = numerator / denometer;
//        println!("tanh decide - t1 {:?}, t2 {:?}, res {:?}, x {:?}", &term1, &term2, &res, x);
        res
    }

    fn gradient(x: f64) -> f64 {
        let t = Self::decide(x); // <Tanh as Function>::decide(x);

        let res = 1.0 - (t*t);
//        println!("tanh derivative - t {:?}, res {:?}, x {:?}", &t, &res, x);
        res
    }
}
