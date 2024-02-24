use ndarray::Array2;
use crate::activation::{Activation, MathFp};

#[derive(Debug)]
pub struct CacheComputation {
    pub z_values: Vec<Array2<f64>>, // linear values
    pub a_values: Vec<Array2<f64>>, // non-linear activation values
    pub funcs: Vec<MathFp>, // activation functions
    pub shapes: Vec<(usize, usize)>, // bias shapes
    pub index: (usize, usize), // function, bias shape
}

impl CacheComputation {
    pub fn new(acts: &[Box<dyn Activation>], biases: &[Array2<f64>]) -> Self {
        // Create activation derivatives collection given activation trait objects
        let funcs: Vec<MathFp> =
            acts.iter().map(|a| { let (_, d) = a.pair(); d}).collect();

        // Compute bias shapes
        let shapes: Vec<(usize, usize)> =
            biases.iter().map(|b| (b.shape()[0], b.shape()[1])).collect();

        CacheComputation {
            z_values: vec![],
            a_values: vec![],
            funcs,
            shapes,
            index: (0, 0),
        }
    }

    // Reset values
    pub fn init(&mut self, x: Array2<f64>) {
        (self.z_values, self.a_values) = (Vec::new(), vec![x]);
        self.index = (self.funcs.len() - 1, self.shapes.len() - 1);
    }

    pub fn cache(&mut self, z: Array2<f64>, a: &Array2<f64>) {
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
        2.0*(&last_a - &y.t())
    }

    pub fn last_a(&mut self) -> Option<Array2<f64>> {
        self.a_values.pop()
    }

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
