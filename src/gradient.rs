use ndarray::{Array2, ArrayView2};
use crate::activation::{ActFp, ActBDFp};
use crate::cost::CostDFp;

pub enum CombinateRule {
    Default(CostDFp, Array2<f64>, ActFp, Array2<f64>),
    CostDotActMatrix(CostDFp, Array2<f64>, ActBDFp, Array2<f64>),
    TermOnly(Array2<f64>),
}

impl CombinateRule {
    pub fn apply(self, f_y: ArrayView2<f64>) -> Array2<f64> { // returns dc_dz
        match self {
            CombinateRule::Default(f, f_a, g, g_z) => {
                let dc_da = (f)(&f_a, &f_y);
                let da_dz = (g)(&g_z);
                dc_da * da_dz
            },
            CombinateRule::TermOnly(v) => v,
            CombinateRule::CostDotActMatrix(f, f_a, g, g_z) => {
                let dc_da = (f)(&f_a, &f_y);
                (g)(dc_da, g_z) // run batch_derivative
            },
        }
    }
}
