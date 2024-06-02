use crate::activation::{ActBDFp, ActFp};
use crate::cost::CostDFp;
use ndarray::{Array2, ArrayView2};

// Simplifies the rules around combining the cost derivative output and activation derivative output
// For example, Sigmoid activation and BinaryCrossEntropy loss derivatives combined are a - y
pub enum CombinateRule {
    Default(
        CostDFp,
        Array2<f64>,
        Option<Array2<f64>>,
        ActFp,
        Array2<f64>,
    ),
    CostDotActMatrix(
        CostDFp,
        Array2<f64>,
        Option<Array2<f64>>,
        ActBDFp,
        Array2<f64>,
    ),
    TermOnly(Array2<f64>),
}

impl CombinateRule {
    // Rust doesn't support currying but think of f_y parameter as a curried param
    pub fn apply(self, f_y: ArrayView2<f64>) -> Array2<f64> {
        // returns dc_dz
        match self {
            CombinateRule::Default(f, f_a, _, g, g_z) => {
                let dc_da = (f)(&f_a, &f_y);
                let da_dz = (g)(&g_z);
                dc_da * da_dz
            }
            CombinateRule::TermOnly(v) => v,
            CombinateRule::CostDotActMatrix(f, f_a, _, g, g_z) => {
                let dc_da = (f)(&f_a, &f_y);
                (g)(dc_da, g_z) // run batch_derivative
            }
        }
    }
}
