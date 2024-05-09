pub mod binary_cross_entropy;
pub mod categorical_cross_entropy;
pub mod quadratic;

use core::f64;
use ndarray::{Array2, ArrayView2};
use std::{convert::From, fmt::Debug};
use std::default::Default;

use crate::gradient::CombinateRule;
use crate::activation::{Act, ActFp};
use crate::cost::{binary_cross_entropy::BinaryCrossEntropy,
                  categorical_cross_entropy::CategoricalCrossEntropy, quadratic::Quadratic}; // Update (2)

pub type CostFp = fn(&Array2<f64>, &Array2<f64>) -> f64; // function pointer
pub type CostDFp = fn(&Array2<f64>, &ArrayView2<f64>) -> Array2<f64>; // function pointer
pub type CostCRFp = fn(CostDFp, Array2<f64>, ArrayView2<f64>, ActFp, Array2<f64>, Act) -> CombinateRule;

pub trait Cost: Debug {
    fn triple(&self) -> (CostFp, CostDFp, CostCRFp);
}

// blanket implementation for all function types T
impl<T: Objective + Debug + Clone + 'static> Cost for T {
    // return both activate, activate derivative func in a tuple
    fn triple(&self) -> (CostFp, CostDFp, CostCRFp) {
        (
            <T as Objective>::evaluate,
            <T as Objective>::derivative,
            <T as Objective>::combinate_rule,
        )
    }
}

pub trait Objective {
    fn evaluate(y: &Array2<f64>, a: &Array2<f64>) -> f64;
    fn derivative(y: &Array2<f64>, a: &ArrayView2<f64>) -> Array2<f64>; // produces dc_da
    // construct relevant derivative combinator rule
    fn combinate_rule(f: CostDFp, f_a: Array2<f64>, _f_y: ArrayView2<f64>, g: ActFp, g_z: Array2<f64>, _act: Act) -> CombinateRule {
        CombinateRule::Default(f, f_a, None, g, g_z)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Loss {
    Quadratic,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
}

impl Default for Loss {
    fn default() -> Self { Loss::Quadratic }
}


impl From<Loss> for Box<dyn Cost> {
    fn from(loss_type: Loss) -> Self {
        // Add new activation type here - Start (3)
        match loss_type {
            Loss::Quadratic => Box::new(Quadratic),
            Loss::BinaryCrossEntropy => Box::new(BinaryCrossEntropy),
            Loss::CategoricalCrossEntropy => Box::new(CategoricalCrossEntropy),
        }
    }
}
