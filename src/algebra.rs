use ndarray::Array2;
use crate::activation::MathFp;

pub trait Algebra<W = Self, B = Self> {
    type Output;

    fn arg_max(&self) -> usize;
    fn weighted_sum(&self, w: &W, b: &B) -> Self::Output;
    fn activate(&self, f: &MathFp) -> Self::Output;
}

impl Algebra for Array2<f64> {
    type Output = Self;

    fn arg_max(&self) -> usize {
        let mut max_acc_index = 0;

        // if we have a single neuron output, return either 0 or 1
        if self.shape() == &[1,1] {
            return self[[0, 0]].round() as usize
        }

        // Find the index of the current neuron with the highest activation
        for (i, &v) in self.iter().enumerate() {
            if v > self[[0, max_acc_index]] { // compare value from first row (0) of 2d array by index
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
    fn activate(&self, f: &MathFp) -> Self {
        self.mapv(|v| f(v))
    }
}
