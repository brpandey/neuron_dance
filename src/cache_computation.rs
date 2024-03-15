use ndarray::{s, Array1, Array2};
use std::collections::HashMap;
use crate::activation::MathFp;

#[derive(Debug)]
pub enum Classification {
    Binary,
    MultiClass,
}

#[derive(Debug)]
pub struct CacheComputation {
    pub z_values: Vec<Array2<f64>>, // linear values
    pub a_values: Vec<Array2<f64>>, // non-linear activation values
    pub funcs: Vec<MathFp>, // activation derivative functions
    pub shapes: Vec<(usize, usize)>, // bias shapes
    pub index: (usize, usize), // function, bias shape
    pub one_hot: HashMap<usize, Array1<f64>>, // store list of one hot encoded vectors
    pub classification: Classification,
    pub output_size: usize,
    pub batch_size: usize,
}

impl CacheComputation {
    pub fn new(backward: &Vec<MathFp>, biases: &[Array2<f64>], output_size: usize, batch_size: usize) -> Self {
        // Compute bias shapes
        let shapes: Vec<(usize, usize)> =
            biases.iter().map(|b| (b.shape()[0], b.shape()[1])).collect();

        // Precompute one hot encoded vectors given output layer size
        let one_hot = Self::precompute(output_size);
        let classification = if output_size > 1 {Classification::MultiClass} else {Classification::Binary};

        CacheComputation {
            z_values: vec![],
            a_values: vec![],
            funcs: backward.clone(), // clone activation derivative functions (back prop)
            shapes,
            index: (0, 0),
            one_hot,
            classification,
            output_size,
            batch_size,
        }
    }

    fn precompute(size: usize) -> HashMap<usize, Array1<f64>> {
        let mut map = HashMap::new();
        let mut zeros;

        for i in 0..size {
            zeros = Array1::zeros(size);
            zeros[i] = 1.;
            map.insert(i, zeros);
        }

        map
    }

    fn one_hot_encode(&self, index: usize) -> Option<&Array1<f64>> {
        self.one_hot.get(&index)
    }

    // Reset values
    pub fn init(&mut self, x: Array2<f64>) {
        (self.z_values, self.a_values) = (Vec::new(), vec![x]);
        self.index = (self.funcs.len() - 1, self.shapes.len() - 1);
    }

    pub fn store(&mut self, z: Array2<f64>, a: &Array2<f64>) {
        self.z_values.push(z);
        self.a_values.push(a.to_owned());
    }

    pub fn nonlinear_derivative(&mut self) -> Option<Array2<f64>>
    {
        if let Some(z_last) = self.last_z() {
            let a_derivative = self.last_func();
            let da_dz = z_last.mapv(|v| a_derivative(v));
            return Some(da_dz)
        }
        None
    }

    /// Assuming cost is (a - y)^2
    pub fn cost_derivative(&mut self, y: &Array2<f64>) -> Array2<f64> {
        let last_a: Array2<f64> = self.last_a().unwrap();

        if let Classification::MultiClass = self.classification {
            // Output labels is a matrix that accounts for output size and mini batch size

            let mut output_labels: Array2<f64> =
                Array2::zeros((self.output_size, self.batch_size));

            // map y to output_labels by:

            // expanding each label value into a one hot encoded value - store result in normalized labels
            // perform for each label in batch

            // e.g. where y is 10 x 1 or 10 x 32
            for i in 0..self.batch_size {
                let label = y[[i, 0]] as usize;
                let encoded_label = self.one_hot_encode(label).unwrap();
                output_labels.slice_mut(s![.., i]).assign(encoded_label);
            }

            &last_a - &output_labels
        } else {
            // e.g. 1 x 1 or 1 x 32
            &last_a - &y.t()
        }
    }

    #[inline]
    pub fn last_a(&mut self) -> Option<Array2<f64>> {
        self.a_values.pop()
    }

    #[inline]
    fn last_z(&mut self) -> Option<Array2<f64>> {
        self.z_values.pop()
    }

    fn last_func(&mut self) -> &MathFp {
        let f = self.funcs.get(self.index.0).unwrap();
        if self.index.0 != 0 { self.index.0 -= 1; }
        f
    }

    pub fn last_bias_shape(&mut self) -> (usize, usize) {
        let s = self.shapes.get(self.index.1).unwrap();
        if self.index.1 != 0 { self.index.1 -= 1; }
        *s
    }
}
