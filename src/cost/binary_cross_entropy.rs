use ndarray::{Array2, ArrayView2};

use crate::activation::{Act, ActFp};
use crate::algebra::AlgebraExt;
use crate::cost::{CostDFp, Objective};
use crate::gradient::CombinateRule;

#[derive(Clone, Debug)]
pub struct BinaryCrossEntropy;

impl Objective for BinaryCrossEntropy {
    fn evaluate(a: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // C = -1/n ∑(y * ln(a)+(1−y)ln(1−a)) or
        // C = 1/n ∑(-y * ln(a)-(1−y)ln(1−a))

        let term1 = -y * (&a.ln());
        let term2 = (1.0 as f64 - y) * (&(1.0 - a).ln());
        let mut diff = term1 - term2;
        diff.mapv_inplace(|v| v.max(0.0)); // replace NaN with 0.0
        diff.mean().unwrap()
    }

    fn derivative(a: &Array2<f64>, y: &ArrayView2<f64>) -> Array2<f64> {
        (a - y) / a * (1.0 - a)
    }

    // override the default trait implementation
    fn combinate_rule(
        f: CostDFp,
        f_a: Array2<f64>,
        _f_y: ArrayView2<f64>,
        g: ActFp,
        g_z: Array2<f64>,
        act: Act,
    ) -> CombinateRule {
        match act {
            //Act::Sigmoid | Act::Sigmoid_(_) => CombinateRule::TermOnly(f_a - f_y), // the shortcut way since terms cancel
            Act::Sigmoid | Act::Sigmoid_(_) => CombinateRule::Default(f, f_a, None, g, g_z),
            _ => CombinateRule::Default(f, f_a, None, g, g_z),
        }
    }
}
