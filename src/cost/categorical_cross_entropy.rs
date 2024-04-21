use ndarray::{Array2, ArrayView2};

use crate::activation::{Act, ActFp, softmax::Softmax};
use crate::algebra::AlgebraExt;
use crate::cost::{Objective, CostDFp};
use crate::gradient::CombinateRule;

#[derive(Clone, Debug)]
pub struct CategoricalCrossEntropy;

const EPSILON: f64 = 1e-20;

impl Objective for CategoricalCrossEntropy {

    fn evaluate(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // C = -1/n ∑(y * ln(a)

        let n = a.shape()[1] as f64;
        let term = -y*&(a + EPSILON).ln(); // replace NaN with 0.0
        term.sum() / n
    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        -y/(a + EPSILON)
    }

    fn combinate_rule(f: CostDFp, f_a: Array2<f64>, _f_y: ArrayView2<f64>, g: ActFp, g_z: Array2<f64>, act: Act) -> CombinateRule {
        match act {
            // Act::Softmax | Act::Softmax_(_) => CombinateRule::TermOnly(f_a - f_y), // the shortcut way since terms cancel
            Act::Softmax | Act::Softmax_(_) => CombinateRule::CostDotActMatrix(f, f_a, Softmax::batch_derivative, g_z),
            _ => CombinateRule::Default(f, f_a, g, g_z),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use crate::cost::Objective;

    #[test]
    fn basic1() {
        let mut a = arr2(&[[0.25,0.25,0.25,0.25], // predictions
                       [0.01,0.01,0.01,0.96]]);

        a.swap_axes(0, 1);

        let mut y = arr2(&[[0.0,0.0,0.0,1.0], // targets
                       [0.0,0.0,0.0,1.0]]);

        y.swap_axes(0, 1);

        let answer = CategoricalCrossEntropy::evaluate(&a, &y);

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

        let answer = CategoricalCrossEntropy::evaluate(&a, &y);

        assert_eq!(answer, 0.47570545188004854);

    }

    #[test]
    fn basic3() {
        let mut a = arr2(&[[0.7, 0.1, 0.2]]);
        a.swap_axes(0, 1);

        let mut y = arr2(&[[1.0, 0.0, 0.0]]);
        y.swap_axes(0, 1);

        let answer = CategoricalCrossEntropy::evaluate(&a, &y);

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
