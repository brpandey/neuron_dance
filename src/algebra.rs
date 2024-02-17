use ndarray::Array2;
use num::Float;
use crate::activation::Activation;

pub trait Algebra<T, W = Self, B = Self> {
    type Output;

    fn arg_max(&self) -> usize;
    fn weighted_sum(&self, w: &W, b: &B) -> Self::Output;
    fn activate(&self, f: &Box<dyn Activation<T>>) -> Self::Output;
}

impl<T: Float + 'static> Algebra<T> for Array2<T> {
    type Output = Self;

    fn arg_max(&self) -> usize {
        let mut max_acc_index = 0;

        // if we have a single neuron output, return either 0 or 1
        if self.shape() == &[1,1] {
            let out = self[[0, 0]].round();
            return T::to_usize(&out).unwrap()
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
    fn activate(&self, f: &Box<dyn Activation<T>>) -> Self {
        self.mapv(|v| f.compute(v))
    }
}
