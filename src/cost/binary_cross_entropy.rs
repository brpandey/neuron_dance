use ndarray::{Array2, ArrayView2};

use crate::activation::Act;
use crate::algebra::AlgebraExt;
use crate::cost::FunctionCost;

#[derive(Clone, Debug)]
pub struct BinaryCrossEntropy;

impl FunctionCost for BinaryCrossEntropy {
    fn compute(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // C = -1/n ∑(y * ln(a)+(1−y)ln(1−a)) or
        // C = 1/n ∑(-y * ln(a)-(1−y)ln(1−a))

        let term1 = -y * (&a.ln());
        let term2 = (1.0 as f64 - y) * (&(1.0 - a).ln());
        let mut diff = term1 - term2;
        diff = diff.mapv(|v| v.max(0.0)); // replace NaN with 0.0
        diff.mean().unwrap()
    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
  //    a - y; 
        (a - y) / a * ( 1.0 - a )
    }

    // override the default trait implementation
    fn combine_derivative(dc_da: Array2<f64>, da_dz: Array2<f64>, _z_last: Array2<f64>, act: Act) -> Array2<f64> {
        // if the output activation is sigmoid then for cross entropy, the da_dz term cancels out
        match act {
            Act::Sigmoid => dc_da * &da_dz, // this can be reduced to (a-y) 
//            Act::Sigmoid => dc_da,
            _ => dc_da * &da_dz,
        }
    }
}
