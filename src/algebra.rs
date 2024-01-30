use ndarray::Array2;

use crate::computation::CacheComputation;
use crate::activation::Activation;

pub fn arg_max(output: &Array2<f64>) -> usize {
    let mut max_acc_index = 0;

    // if we have a single neuron output, return either 0 or 1
    if output.shape() == &[1,1] {
        return output[[0, 0]].round() as usize
    }

    // Find the index of the output neuron with the highest activation
    for (i, &v) in output.iter().enumerate() {
        if v > output[[0, max_acc_index]] { // compare value from first row (0) of 2d array by index
            max_acc_index = i;
        }
    }

    max_acc_index
}

// Z = W*X + B
#[inline]
pub fn z_linear(w: &Array2<f64>, x: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    w.dot(x) + b
}

#[inline]
pub fn a_nonlinear(z: &mut Array2<f64>, f: &Box<dyn Activation>) -> Array2<f64> {
    z.mapv(|v| f.compute(v))
}

pub fn nonlinear_derivative(cc: &mut CacheComputation) -> Option<Array2<f64>>
{
    if let (Some(z_last), Some(f)) = (cc.last_z(), cc.last_func()) {
        let da_dz = z_last.mapv(|v| f.derivative(v));
        return Some(da_dz)
    }
    None
}

/// Assuming cost is (a - y)^2
pub fn cost_derivative(cc: &mut CacheComputation, y: &Array2<f64>) -> Array2<f64> {
    let output_a: Array2<f64> = cc.last_a().unwrap();
    2.0*(&output_a - y)
}
