use crate::activation::Decider;
use crate::algebra::AlgebraExt;
use ndarray::{Array2, Axis};

#[derive(Clone, Debug)]
pub struct Softmax;

impl Decider for Softmax {
    // Softmax requires vectorized implementations of activate and derivative
    // Since it looks for max and sum values

    // Formula
    // expd = e^(x - max(x))
    // expd / expd.sum(axis=0)

    fn activate(z: &Array2<f64>) -> Array2<f64> {
        let max = z.max_axis(Axis(0)); //.into_shape((1, cols)).unwrap();
        let expd = (z - &max).exp();
		    &expd / &expd.sum_axis(Axis(0))
    }

    // Assumes z only grabs one row in Array2, the first row
    fn derivative(z: &Array2<f64>) -> Array2<f64> {
        let soft = Self::activate(z);
        let s = soft.index_axis(Axis(1), 0);

        // square matrix, values on central diagonal
        let mut jacobian_matrix = Array2::from_diag(&s);
        let mut kronecker_ij; // way to encapsulate if-else logic, for constants 0 and 1, making assign clause tidy

        // iterative method

        let side = jacobian_matrix.shape()[0];
        for i in 0..side {
            for j in 0..side {
                kronecker_ij = if i == j { 1.0 } else { 0.0 };
                jacobian_matrix[(i, j)] = s[j] * (kronecker_ij - s[i]);
            }
        }

        jacobian_matrix
    }
}

impl Softmax {
    pub fn batch_derivative(dc_da: Array2<f64>, _da_dz: Array2<f64>, z_last: Array2<f64>) -> Array2<f64> {
        let z_iter = z_last.axis_iter(Axis(1)); // grab z values by column
        let c_iter = dc_da.axis_iter(Axis(1)); // grab c values by column
        let output_size = dc_da.shape()[0];
        let mut dc_dz = Array2::zeros((output_size, 0));

        // For each col this calc is being down and then pushed back into aggregate acc dc_dz
        // dc_dz = dc_da dot da_dz

		// Compute batch dc_dz based on taking the individual cost gradient and z single values
        dc_dz = c_iter.zip(z_iter).fold(dc_dz, |mut acc, (cost_grad_single, z_col)| {
            let z_single = z_col.to_owned().into_shape((output_size, 1)).unwrap(); // put z into array2
            let softmax_grad_single = Softmax::derivative(&z_single); // get da_dz single
            let dc_dz_single = softmax_grad_single.dot(&cost_grad_single); // e.g. da_dz - jacobian matrix (10x10) dot 10x1 dc_da
            let _ = acc.push_column(dc_dz_single.view());
            acc
        });

        dc_dz // dc_da * da_dz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use crate::activation::Decider;

    #[test]
    fn basic1() {
        let mut z = arr2(&[[1.0, 2.0]]); // predictions
		z.swap_axes(0,1);

        let answer = Softmax::activate(&z);

        assert_eq!(answer, arr2(&[[0.2689414213699951],
                                  [0.7310585786300049]]));

        let grad = Softmax::derivative(&z);

        assert_eq!(grad, arr2(&[[0.19661193324148185, -0.19661193324148185],
                                 [-0.19661193324148185, 0.19661193324148185]]));

    }

    #[test]
    fn basic2() {
        let mut z = arr2(&[[0.354, 0.418, 0.482, 0.546, 0.16]]);
        z.swap_axes(0,1);

        let answer = Softmax::activate(&z);

        assert_eq!(answer, arr2(&[[0.19091351615156651],
                                  [0.20353144839001694],
                                  [0.21698333003751666],
                                  [0.23132427881095552],
                                  [0.15724742660994437]]));

        let grad = Softmax::derivative(&z);

        assert_eq!(grad, arr2(&[[0.15446554550221206, -0.03885690445955923, -0.041425050483738124, -0.04416293143902483, -0.030020659119889884],
                                 [-0.03885690445955923, 0.16210639790627882, -0.04416293143902483, -0.047081765514169885, -0.03200479649352487],
                                 [-0.041425050483738124, -0.04416293143902483, 0.16990156452334676, -0.05019351233492808, -0.034120070265655736],
                                 [-0.04416293143902483, -0.047081765514169885, -0.05019351233492808, 0.17781335684354682, -0.03637514755542404],
                                 [-0.030020659119889884, -0.03200479649352487, -0.034120070265655736, -0.03637514755542404, 0.13252067343449453]]));
    }
}
