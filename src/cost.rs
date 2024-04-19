pub mod binary_cross_entropy;
pub mod categorical_cross_entropy;
pub mod quadratic;

use ndarray::{Array2, ArrayView2};
use core::f64;
use std::{convert::From, fmt::Debug};

use crate::activation::Act;
use crate::cost::{binary_cross_entropy::BinaryCrossEntropy,
                  categorical_cross_entropy::CategoricalCrossEntropy, quadratic::Quadratic}; // Update (2)

pub type CostFp = fn(&Array2<f64>, &Array2<f64>) -> f64; // function pointer
pub type CostDFp = fn(&Array2<f64>, &ArrayView2<f64>) -> Array2<f64>; // function pointer
pub type CostCDFp = fn(Array2<f64>, Array2<f64>, Array2<f64>, Act) -> Array2<f64>; // function pointer

pub trait Cost: Debug {
    fn triple(&self) -> (CostFp, CostDFp, CostCDFp);
}

// blanket implementation for all function types T
impl<T: FunctionCost + Debug + Clone + 'static> Cost for T {
    // return both activate, activate derivative func in a tuple
    fn triple(&self) -> (CostFp, CostDFp, CostCDFp) {
        (
            <T as FunctionCost>::compute,
            <T as FunctionCost>::derivative,
            <T as FunctionCost>::combine_derivative,
        )
    }
}

pub trait FunctionCost {
    fn compute(y: &Array2<f64>, a: &Array2<f64>) -> f64;
    fn derivative(y: &Array2<f64>, a: &ArrayView2<f64>) -> Array2<f64>; // produces dc_da
    fn combine_derivative(dc_da: Array2<f64>, da_dz: Array2<f64>, _z_last: Array2<f64>, _act: Act) -> Array2<f64> {
            dc_da * da_dz
    }
}

#[derive(Copy, Clone)]
pub enum Loss {
    Quadratic,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
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
