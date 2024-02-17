use std::collections::VecDeque;
use std::ops::MulAssign;
use ndarray::{Array2, ScalarOperand};
use num::Float;

use crate::activation::{Activation, MathFp};

#[derive(Debug)]
pub struct CacheComputation<T> {
    z_values: Vec<Array2<T>>, // linear values
    a_values: Vec<Array2<T>>, // non-linear activation values
    funcs: Vec<MathFp<T>>,
    lastf: usize,
}

impl<T: Float + MulAssign + ScalarOperand> CacheComputation<T> {
    pub fn new(acts: &[Box<dyn Activation<T>>]) -> Self {
        // Create activation derivatives collection given activation trait objects
        let funcs: Vec<MathFp<T>> =
            acts.iter().map(|a| { let (_, d) = a.pair(); d}).collect();

        CacheComputation {
            z_values: vec![],
            a_values: vec![],
            funcs,
            lastf: 0,
        }
    }

    pub fn init(&mut self, x: Array2<T>) {
        (self.z_values, self.a_values) = (Vec::new(), vec![x]);
        self.lastf = self.funcs.len() - 1;
    }

    pub fn cache(&mut self, z: Array2<T>, a: &Array2<T>) {
        self.z_values.push(z);
        self.a_values.push(a.to_owned());
    }

    fn nonlinear_derivative(&mut self) -> Option<Array2<T>>
    {
        if let (Some(z_last), Some(a_derivative)) = (self.last_z(), self.last_func()) {
            let da_dz = z_last.mapv(|v| a_derivative(v));
            return Some(da_dz)
        }
        None
    }

    /// Assuming cost is (a - y)^2
    fn cost_derivative(&mut self, y: &Array2<T>) -> Array2<T> {
        let output_a: Array2<T> = self.last_a().unwrap();
        (&output_a - y) * T::from(2.0).unwrap()
    }

    #[inline]
    fn last_a(&mut self) -> Option<Array2<T>> {
        self.a_values.pop()
    }

    #[inline]
    fn last_z(&mut self) -> Option<Array2<T>> {
        self.z_values.pop()
    }

    fn last_func(&mut self) -> Option<&MathFp<T>> {
        let f = self.funcs.get(self.lastf);
        if self.lastf != 0 { self.lastf -= 1; }
        f
    }
}

#[derive(Debug)]
pub struct ChainRuleComputation<'a, T> {
    cache: &'a mut CacheComputation<T>,
    bias_deltas: VecDeque<Array2<T>>,
    weight_deltas: VecDeque<Array2<T>>,
}

impl <'a, T: Float + MulAssign + ScalarOperand> ChainRuleComputation<'a, T> {
    pub fn new(cache: &'a mut CacheComputation<T>) -> Self {
        ChainRuleComputation {
            cache,
            bias_deltas: VecDeque::new(),
            weight_deltas: VecDeque::new(),
        }
    }

    #[inline]
    pub fn bias_deltas(&self) -> impl Iterator<Item = &'_ Array2<T>> { // iterator is tied to the lifetime of current computation
        self.bias_deltas.iter()
    }

    #[inline]
    pub fn weight_deltas(&self) -> impl Iterator<Item = &'_ Array2<T>> { // iterator is tied to the lifetime of current computation
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

    pub fn init(&mut self, y: &Array2<T>) -> Array2<T> {
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
        let dc_dz2 = dc_da2.dot(&da2_dz2);

        // Z2 = W2A1 + B, dz_db is just the constant 1 that is multiplying B
        let dz2_db2 = T::from(1.0).unwrap();
        let dc_db2 = &dc_dz2 * dz2_db2;

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
    pub fn fold_layer(&mut self, dc_dz2: Array2<T>, w: &Array2<T>) -> Array2<T>{
        // Main equations
        // dc_db = dc_dz2 * dz2_da1 * da1_dz1 * dz1_db1   (or) dc_dz1 * dz1_db1
        // dc_dw = dc_dz2 * dz2_da1 * da1_dz1 * dz1_dw1.t (or) dc_dz1 * dz1_dw1.t

        // Z2 = W2A1 + B, w is just W2
        let dz2_da1 = w;

        // For example: A1 = Relu(Z1)
        let da1_dz1 = self.cache.nonlinear_derivative().unwrap(); // derivative of relu applied to Z1
        let dc_dz1 = dc_dz2.dot(dz2_da1).dot(&da1_dz1); // dc_dz is the accumulator value that allows us to repeatedly call fold layer

        // For example Z1 = W1X + B1
        let dz1_db1 = T::from(1.0).unwrap();
        let dc_db1 = &dc_dz1 * dz1_db1; // cost derivative with respect to bias

        let dz1_dw1 = self.cache.last_a().unwrap(); // if 2 layer nn, will be X
        let dc_dw1 = dc_dz1.dot(&dz1_dw1.t()); // cost derivative with respect to weight

        self.bias_deltas.push_front(dc_db1); // fold result into deque
        self.weight_deltas.push_front(dc_dw1); // fold result into deque

        dc_dz1 // return acc
    }
}
