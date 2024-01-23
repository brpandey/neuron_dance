use std::collections::VecDeque;
use ndarray::Array2;

use crate::activation::Function;
use crate::algebra::{cost_derivative, apply_nonlinear_derivative};

pub struct CacheComputation {
    pub z_values: Vec<Array2<f64>>, // linear values
    pub a_values: Vec<Array2<f64>>, // non-linear activation values
    pub func_names: Vec<Function>,
    pub currentf: usize,
}

impl CacheComputation {
    pub fn new(func_names: Vec<Function>) -> Self {
        let (z_values, a_values) = (Vec::new(), Vec::new());

        CacheComputation {
            z_values,
            a_values,
            func_names,
            currentf: 0,
        }
    }

    pub fn init(&mut self, x: Array2<f64>) {
        (self.z_values, self.a_values) = (Vec::new(), vec![x]);
        self.currentf = self.func_names.len() - 1;
    }

    pub fn store_intermediate(&mut self, z: Array2<f64>, a: &Array2<f64>) {
        self.z_values.push(z);
        self.a_values.push(a.to_owned());
    }

    pub fn last_a(&mut self) -> Option<Array2<f64>> {
        self.a_values.pop()
    }

    pub fn last_z(&mut self) -> Option<Array2<f64>> {
        self.z_values.pop()
    }

    pub fn last_func(&mut self) -> Option<&Function> {
        if self.currentf != 0 { self.currentf -= 1; }
        self.func_names.get(self.currentf)
    }
}

pub struct ChainRuleComputation<'a> {
    pub cache: &'a mut CacheComputation,
    pub bias_deltas: VecDeque<Array2<f64>>,
    pub weight_deltas: VecDeque<Array2<f64>>,
}

impl <'a> ChainRuleComputation<'a> {
    pub fn new(cache: &'a mut CacheComputation) -> Self {
        ChainRuleComputation {
            cache,
            bias_deltas: VecDeque::new(),
            weight_deltas: VecDeque::new(),
        }
    }

    pub fn bias_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> { // iterator is tied to the lifetime of current computation
        self.bias_deltas.iter()
    }

    pub fn weight_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> { // iterator is tied to the lifetime of current computation
        self.weight_deltas.iter()
    }

    // compute chain rule for last/output layer
    pub fn init(&mut self, y: &Array2<f64>) -> Array2<f64> {
        // dc_db => dc_da * da_dz * dz_db

        // Here are the functions in an example 2 layer NN with relu then sigmoid activation
        // C = (A2 − Y )
        let dc_da = cost_derivative(self.cache, y); // cost derivative wrt to last activation layer

        // A2 = sigmoid (Z2)
        let da_dz = apply_nonlinear_derivative(self.cache).unwrap();
        let dc_dz: Array2<f64> =  dc_da * da_dz;

        // Z2 = W2A1 + B, dz_db is just the constant 1 that is multiplying B
        let dz_db = 1.0;
        let dc_db = &dc_dz * dz_db;
        self.bias_deltas.push_front(dc_db);

        // dc_dw => dc_dz * dz_dw  or also  dc_da * da_dz * dz_dw

        // Z2 = W2A1 + B, dz_dw is just the constant A1 which is multiplying W2 
        let dz_dw = self.cache.last_a().unwrap();
        let dc_dw = dc_dz.dot(&dz_dw.t());
        self.weight_deltas.push_front(dc_dw);

        dc_dz
    }

    // computes chain rule for preceding layer <starting from next to last layer>
    pub fn fold_layer(&mut self, mut dc_dz: Array2<f64>, w: &Array2<f64>) -> Array2<f64>{
        // Note j is layer after the layer i, e.g. j is 2 and i is 1

        let dzj_dai = w.t();
        let dai_dzi = apply_nonlinear_derivative(self.cache).unwrap();

        // (dZ2_dA1 dot dC_dZ2) * dA1_dZ1 => dC_dZ1 or
        // (dzj_dai dot dC_dZj) * dai_dzi => dC_dZi where j is the layer after i
        dc_dz = dzj_dai.dot(&dc_dz) * dai_dzi;

        let dz_db = 1.0;
        let dc_db = &dc_dz * dz_db; // cost derivative with respect to bias
        self.bias_deltas.push_front(dc_db); // fold result into deque

        let dz_dw = self.cache.last_a().unwrap();
        let dc_dw = dc_dz.dot(&dz_dw.t()); // cost derivative with respect to weight
        self.weight_deltas.push_front(dc_dw); // fold result into deque

        dc_dz // return acc
    }
}
