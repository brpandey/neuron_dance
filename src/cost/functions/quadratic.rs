use ndarray::{Array2, ArrayView2};

use crate::cost::functions::FunctionCost;

#[derive(Clone, Debug)]
pub struct Quadratic;

impl FunctionCost for Quadratic {
    fn compute(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let sub = a - y;
        let sq = &sub * &sub * 0.5;
        let res = sq.sum(); //mean().unwrap();

 //       println!("Quadratic cost, result {:?}", &res);
        res

    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        a - y
    }

}
