use ndarray::{Array2, ArrayView2};

use crate::cost::FunctionCost;
use crate::algebra::AlgebraExt;
use crate::activation::functions::Act;

#[derive(Clone, Debug)]
pub struct CrossEntropy;

impl FunctionCost for CrossEntropy {

    fn compute(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // C = -1/n ∑(y * ln(a)+(1−y)ln(1−a))
        // or
        // C = 1/n ∑(-y * ln(a)-(1−y)ln(1−a))

        let term1 = -y*(&a.ln());
        let term2 = (1.0 as f64 - y)*(&(1.0 - a).ln());
        let diff = term1 - term2;
        let res = diff.mean().unwrap();

//        println!("Cross entropy cost, result {:?}", &res);

        res
    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        a - y
    }

    // override the default trait implementation
    fn combine_derivative(dc_da: Array2<f64>, da_dz: Array2<f64>, act: Act) -> Array2<f64> {
        // if the output activation is sigmoid then for cross entropy, the da_dz term cancels out
        if let Act::Sigmoid = act {
            dc_da
        } else {
            dc_da * &da_dz
        }
    }

}
