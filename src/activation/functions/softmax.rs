use ndarray::{Array2, Axis};
use crate::activation::functions::FunctionAct;
use crate::algebra::AlgebraExt;

#[derive(Clone, Debug)]
pub struct Softmax;

impl FunctionAct for Softmax {

    // Softmax requires vectorized implementations of activate and derivative
    // Since it looks for max and sum values

    // Formula
    // expd = e^(x - max(x))
    // expd / expd.sum(axis=0)

    fn activate(x: &Array2<f64>) -> Array2<f64> {
        let max = x.max_axis(Axis(1));
        let expd = (x - max).exp();
        let ret = &expd / &expd.sum_axis(Axis(1));
        ret
    }

    fn derivative(x: &Array2<f64>) -> Array2<f64> {
        let s = Self::activate(x);

        // square matrix, values on central diagonal
        let mut jacobian_matrix = Array2::from_diag(&s.index_axis(Axis(0), 0)); // grab first row

        // iterative method
        let side = jacobian_matrix.shape()[0];
        for i in 0..side {
            for j in 0..side {
                if i == j { // if on a diagonal
                    jacobian_matrix[(i,j)] = s[(0, i)] * (1.0 - s[(0,i)]);
                } else {
                    jacobian_matrix[(i,j)] = -s[(0, i)] * s[(0, j)];
                }
            }
        }

        jacobian_matrix
    }
}
