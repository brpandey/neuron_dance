use ndarray::{Array2};
use crate::activation::Function;

pub struct ForwardComputation {
    pub z_values: Vec<Array2<f64>>, // linear values
    pub a_values: Vec<Array2<f64>>, // non-linear activation values
    pub func_names: Vec<Function>,
    pub current_func_i: usize,
}


impl ForwardComputation {
    pub fn new(x: Array2<f64>, func_names: Vec<Function>) -> Self {
        let (z_values, a_values) = (Vec::new(), vec![x]);
        let func_size = func_names.len();

        ForwardComputation {
            z_values,
            a_values,
            func_names,
            current_func_i: func_size,
        }
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
        if self.current_func_i != 0 { self.current_func_i -= 1; }
        self.func_names.get(self.current_func_i)
    }
}

