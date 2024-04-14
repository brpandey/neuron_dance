use ndarray::{Array2, ArrayView2};

use crate::algebra::AlgebraExt;
use crate::cost::FunctionCost;

#[derive(Clone, Debug)]
pub struct Quadratic;

impl FunctionCost for Quadratic {
    fn compute(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let sub = (a - y).normalize();
        let sq = &sub * &sub * 0.5;
//        println!("Quadratic cost, result {:?}", &sq);
        sq
    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        a - y
    }

}
