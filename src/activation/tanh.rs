use crate::activation::Decider;

#[derive(Clone, Debug)]
pub struct Tanh;

impl Decider for Tanh {
    fn decide(z: f64) -> f64 {
        let t1 = (z).exp() - (-z).exp();
        let t2 = (z).exp() + (-z).exp();

        let res = t1 / t2;
//        println!("tanh decide - t1 {:?}, t2 {:?}, res {:?}, z {:?}", &term1, &term2, &res, z);
        res
    }

    fn gradient(z: f64) -> f64 {
        let t = Self::decide(z); // <Tanh as Function>::decide(z);

        let res = 1.0 - (t*t);
//        println!("tanh derivative - t {:?}, res {:?}, z {:?}", &t, &res, z);
        res
    }
}
