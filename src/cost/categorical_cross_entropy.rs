use ndarray::{Array2, ArrayView2};

use crate::activation::{Act, softmax::Softmax};
use crate::algebra::AlgebraExt;
use crate::cost::FunctionCost;

#[derive(Clone, Debug)]
pub struct CategoricalCrossEntropy;

const EPSILON: f64 = 1e-20;

impl FunctionCost for CategoricalCrossEntropy {

    fn compute(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // C = -1/n ∑(y * ln(a)

        let n = a.shape()[1] as f64;
        let term = -y*&(a + EPSILON).ln(); // replace NaN with 0.0
        term.sum() / n
    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
//        let res = a - y; 
        -y/(a + EPSILON)
    }

    // override the default trait implementation
    fn combine_derivative(dc_da: Array2<f64>, da_dz: Array2<f64>, z_last: Array2<f64>, act: Act) -> Array2<f64> {
        // if the output activation is softmax then for categorical cross entropy, the da_dz term cancels out
        let ret = match act {
            Act::Sigmoid | Act::SigmoidW(_) => dc_da * &da_dz,
            //Act::Softmax | Act::Softmax_(_) => dc_da,
            Act::Softmax | Act::Softmax_(_) => Softmax::batch_derivative(dc_da, da_dz, z_last),
            _ => dc_da * &da_dz,
        };

        ret
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use crate::cost::FunctionCost;

    #[test]
    fn basic1() {
        let mut a = arr2(&[[0.25,0.25,0.25,0.25], // predictions
                       [0.01,0.01,0.01,0.96]]);

        a.swap_axes(0, 1);

        let mut y = arr2(&[[0.0,0.0,0.0,1.0], // targets
                       [0.0,0.0,0.0,1.0]]);

        y.swap_axes(0, 1);

        let answer = CategoricalCrossEntropy::compute(&a, &y);

        assert_eq!(answer, 0.7135581778200729);
    }

    #[test]
    fn basic2() {
        let mut a = arr2(&[[0.5, 0.3, 0.1, 0.1],
                       [0.2, 0.1, 0.6, 0.1],
                       [0.1, 0.8, 0.05, 0.05]]);

        a.swap_axes(0, 1);

        let mut y = arr2(&[[1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0]]);

        y.swap_axes(0, 1);

        let answer = CategoricalCrossEntropy::compute(&a, &y);

        assert_eq!(answer, 0.47570545188004854);

    }

    #[test]
    fn basic3() {
        let mut a = arr2(&[[0.7, 0.1, 0.2]]);
        a.swap_axes(0, 1);

        let mut y = arr2(&[[1.0, 0.0, 0.0]]);
        y.swap_axes(0, 1);

        let answer = CategoricalCrossEntropy::compute(&a, &y);

        assert_eq!(answer, 0.35667494393873245);

    }

    #[test]
    fn basic4() {
        let mut a = arr2(&[[0.05, 0.85, 0.10, 0.0]]);
        let mut y = arr2(&[[0.0, 1.0, 0.0, 0.0]]);

        a.swap_axes(0,1);
        y.swap_axes(0,1);

        let answer = CategoricalCrossEntropy::derivative(&a, &y.view());

        assert_eq!(answer, arr2(&[[0.0],
                                  [-1.1764705882352942],
                                  [0.0],
                                  [0.0]]));
    }
}
