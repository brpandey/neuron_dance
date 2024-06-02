use ndarray::Array2;
/// Chain rule
/// Aggregates gradients of each layer into a reduction
use std::collections::VecDeque;

use crate::chain_layer::{ComputeLayer, HiddenLayerTerms, OutputLayerTerms, SharedHiddenTerms};
use crate::gradient_cache::GradientCache;
use crate::gradient_stack::GT;

pub trait ChainRule {
    fn chain_rule(&mut self) -> (Array2<f64>, Array2<f64>);
}

#[derive(Debug)]
pub struct ChainRuleComputation<'a> {
    pub gc: &'a mut GradientCache,
    pub bias_deltas: VecDeque<Array2<f64>>,
    pub weight_deltas: VecDeque<Array2<f64>>,
}

impl<'a> ChainRuleComputation<'a> {
    pub fn new(gc: &'a mut GradientCache) -> Self {
        ChainRuleComputation {
            gc,
            weight_deltas: VecDeque::new(),
            bias_deltas: VecDeque::new(),
        }
    }

    pub fn bias_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> {
        // iterator is tied to the lifetime of current computation
        self.bias_deltas.iter()
    }

    pub fn weight_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> {
        // iterator is tied to the lifetime of current computation
        self.weight_deltas.iter()
    }

    // Compute chain rule for last/output layer
    // For example purposes consider a 3 layer NN including an input layer
    // with these equations:

    /*
       C  = (A2 - Y)^2
       A2 = sigmoid(Z2)
       Z2 = W2*A1 + B2
       A1 = relu(Z1)
       Z1 = W1*X + B1

       Where * is the dot product of two matrices l*m and m*n resulting in a matrix l*n
    */

    pub fn init(&mut self, y: &Array2<f64>) -> Array2<f64> {
        // Main chain rule equations
        // dc_db => dc_da * da_dz * dz_db   (or) dc_dz * dz_db
        // dc_dw => dc_da * da_dz * dz_dw.t (or) dc_dz * dz_dw.t

        // e.g. C = (A2 − Y)^2, A2 = sigmoid (Z2)
        // create current layer's terms
        let mut layer_terms = ComputeLayer::Output(OutputLayerTerms {
            dc_dz: Some(self.gc.cost_derivative(y)),
            dz_db: 1.0,
            dz_dw: self.gc.stack.pop(GT::Nonlinear).array(),
            bias_shape: self.gc.stack.pop(GT::BiasShape).shape(),
        });

        let results = layer_terms.chain_rule();

        self.bias_deltas.push_front(results.0); // fold result into deque
        self.weight_deltas.push_front(results.1); // fold result into deque

        layer_terms.acc().unwrap()
    }

    // Method can be called repeatedly:
    // Computes chain rule from preceding layer value (layer j) and calculates for current layer (layer i)
    // With layer j being layer after layer i in a feed forward nn, so e.g. j is 2 and i 1
    // Bias and weight delta values are folded into the respective collections
    pub fn fold_layer(&mut self, dc_dz2: Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
        // Main equations
        // dc_db = (dc_dz2 * dz2_da1 * da1_dz1) * dz1_db1   (or) dc_dz1 * dz1_db1
        // dc_dw = (dc_dz2 * dz2_da1 * da1_dz1) * dz1_dw1.t (or) dc_dz1 * dz1_dw1.t

        // create shared component
        let shared = SharedHiddenTerms {
            dc_dz2,
            dz2_da1: w.clone(), // Z2 = W2A1 + B, w is just W2
            da1_dz1: self.gc.activation_derivative(), // derivative of e.g. relu applied to Z1,
            dc_dz1: None,
        }; // last field is result

        // current layer's terms
        let mut layer_terms = ComputeLayer::Hidden(HiddenLayerTerms {
            shared,
            dz1_db1: 1.0, // For example Z1 = W1X + B1
            dz1_dw1: self.gc.stack.pop(GT::Nonlinear).array(),
            bias_shape: self.gc.stack.pop(GT::BiasShape).shape(),
        });

        let results = layer_terms.chain_rule();

        self.bias_deltas.push_front(results.0); // fold result into deque
        self.weight_deltas.push_front(results.1); // fold result into deque

        layer_terms.acc().unwrap() // return acc
    }
}
