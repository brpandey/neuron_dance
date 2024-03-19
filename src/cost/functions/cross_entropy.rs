use ndarray::{Array2, ArrayView2};

use crate::cost::FunctionCost;
use crate::algebra::AlgebraExt;

#[derive(Clone, Debug)]
pub struct CrossEntropy;

impl FunctionCost for CrossEntropy {

    fn compute(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // C = -1/n ∑(y * ln(a)+(1−y)ln(1−a))
        println!("Cross entropy cost, compute {:?} {:?}", &a, &y);

        let term1 = y.t().dot(&a.ln());
        let term2 = (1.0 as f64 - y.t().to_owned()).dot(&(1.0 - a).ln());
        let sum = term1 + term2;

        let res = -1.0*sum.mean().unwrap();

        println!("Cross entropy cost, result {:?}", &res);

        res
    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        a - y
    }
}
