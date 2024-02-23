use std::collections::VecDeque;
use ndarray::{Array2, Axis};
use crate::cache_computation::CacheComputation;

#[derive(Debug)]
pub struct ChainRuleComputation<'a> {
    pub cache: &'a mut CacheComputation,
    pub bias_deltas: VecDeque<Array2<f64>>,
    pub weight_deltas: VecDeque<Array2<f64>>,
}

impl <'a> ChainRuleComputation<'a> {
    pub fn new(cache: &'a mut CacheComputation) -> Self {
        ChainRuleComputation {
            cache,
            weight_deltas: VecDeque::new(),
            bias_deltas: VecDeque::new(),
        }
    }

    pub fn bias_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> { // iterator is tied to the lifetime of current computation
        self.bias_deltas.iter()
    }

    pub fn weight_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> { // iterator is tied to the lifetime of current computation
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
        // C = (A2 - Y)^2
        // A2 = sigmoid(Z2)

        // Main chain rule equations
        // dc_db => dc_da * da_dz * dz_db   (or) dc_dz * dz_db
        // dc_dw => dc_da * da_dz * dz_dw.t (or) dc_dz * dz_dw.t

        // Here are the functions in an example 2 layer NN with relu then sigmoid activation
        // For example, C = (A2 − Y)^2
        let dc_da2 = self.cache.cost_derivative(y); // cost derivative wrt to last activation layer

        // A2 = sigmoid (Z2)
        let da2_dz2 = self.cache.nonlinear_derivative().unwrap();
        let dc_dz2: Array2<f64> = &dc_da2 * &da2_dz2;

        // Z2 = W2A1 + B, dz_db is just the constant 1 that is multiplying B
        let dz2_db2 = 1.0;
        let dc_db2_temp = (&dc_dz2 * dz2_db2).sum_axis(Axis(1));
        let dc_db2 = dc_db2_temp.into_shape(self.cache.last_bias_shape()).unwrap();

        // Z2 = W2A1 + B, dz_dw is just the constant A1 which is multiplying W2
        let dz2_dw2 = self.cache.last_a().unwrap();
        let dc_dw2 = dc_dz2.dot(&dz2_dw2.t());

        self.bias_deltas.push_front(dc_db2);
        self.weight_deltas.push_front(dc_dw2);

        dc_dz2
    }

    // Method can be called repeatedly:
    // Computes chain rule from preceding layer value (layer 2) and calculates for current layer (layer 1)
    // With layer j being layer after layer i in a feed forward nn, so e.g. j is 2 and i 1
    pub fn fold_layer(&mut self, dc_dz2: Array2<f64>, w: &Array2<f64>) -> Array2<f64>{
        // Main equations
        // dc_db = dc_dz2 * dz2_da1 * da1_dz1 * dz1_db1   (or) dc_dz1 * dz1_db1
        // dc_dw = dc_dz2 * dz2_da1 * da1_dz1 * dz1_dw1.t (or) dc_dz1 * dz1_dw1.t

        // Z2 = W2A1 + B, w is just W2
        let dz2_da1 = w.t();

        // For example: A1 = Relu(Z1)
        let da1_dz1 = self.cache.nonlinear_derivative().unwrap(); // derivative of relu applied to Z1
        let dc_da1 = dz2_da1.dot(&dc_dz2);
        let dc_dz1 = dc_da1 * da1_dz1;

        // For example Z1 = W1X + B1
        let dz1_db1 = 1.0;
        let dc_db1_temp = (&dc_dz1 * dz1_db1).sum_axis(Axis(1));
        let dc_db1 = dc_db1_temp.into_shape(self.cache.last_bias_shape()).unwrap(); // cost derivative with respect to bias

        let dz1_dw1 = self.cache.last_a().unwrap();
        let dc_dw1 = dc_dz1.dot(&dz1_dw1.t()); // cost derivative with respect to weight

        self.bias_deltas.push_front(dc_db1); // fold result into deque
        self.weight_deltas.push_front(dc_dw1); // fold result into deque

        dc_dz1 // return acc
    }
}
