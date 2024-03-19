use ndarray::Array2;
use crate::activation::ActFp;

pub trait AlgebraExt<W = Self, B = Self> {
    type Output;

    fn arg_max(&self) -> usize;
    fn weighted_sum(&self, w: &W, b: &B) -> Self::Output;
    fn activate(&self, f: &ActFp) -> Self::Output;
    fn ln(&self) -> Self::Output;
}

impl AlgebraExt for Array2<f64> {
    type Output = Self;

    fn arg_max(&self) -> usize {
        let mut max_acc_index = 0;

        // if we have a single neuron output, return either 0 or 1
        if self.shape() == &[1,1] {
            return self[[0, 0]].round() as usize
        }

        // Find the index of the current neuron with the highest activation
        for (i, &v) in self.iter().enumerate() {
            if v > self[[max_acc_index, 0]] { // compare value from first column (0) of 2d array by index
                max_acc_index = i;
            }
        }

        max_acc_index
    }

    // Z = W*X + B
    #[inline]
    fn weighted_sum(&self, w: &Self, b: &Self) -> Self {
        w.dot(self) + b
    }

    #[inline]
    // perform non-linear activation
    fn activate(&self, f: &ActFp) -> Self {
        self.mapv(|v| f(v))
    }

    #[inline]
    // perform natural logarithm - ln
    fn ln(&self) -> Self {
        self.mapv(|v| v.log(std::f64::consts::E))
    }

}
