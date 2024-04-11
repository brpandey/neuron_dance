use ndarray::{Array1, Array2, Axis};

pub trait AlgebraExt<W = Self, B = Self> {
    type Output;
    type Output1;

    fn arg_max(&self) -> usize;
    fn weighted_sum(&self, w: &W, b: &B) -> Self::Output;
    fn exp(&self) -> Self::Output;
    fn ln(&self) -> Self::Output;
    fn normalize(&self) -> f64;
    fn min_axis(&self, axis: Axis) -> Self::Output1;
    fn max_axis(&self, axis: Axis) -> Self::Output1;
    fn sqrt(&self) -> Self::Output;
}

impl AlgebraExt for Array2<f64> {
    type Output = Self;
    type Output1 = Array1<f64>;

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
    fn weighted_sum(&self, w: &Self, b: &Self) -> Self::Output {
        w.dot(self) + b
    }

    #[inline]
    // perform natural logarithm - ln
    fn ln(&self) -> Self::Output {
        self.mapv(|v| v.log(std::f64::consts::E))
    }

    #[inline]
    fn normalize(&self) -> f64 {
        (self*self).sum().sqrt()
    }

    #[inline]
    fn min_axis(&self, axis: Axis) -> Self::Output1 {
        self.map_axis(axis, |v| *v.iter().min_by(|a,b| a.total_cmp(b)).unwrap())
    }

    #[inline]
    fn max_axis(&self, axis: Axis) -> Self::Output1 {
        self.map_axis(axis, |v| *v.iter().max_by(|a,b| a.total_cmp(b)).unwrap())
    }

    #[inline]
    fn sqrt(&self) -> Self::Output {
        self.mapv(|v| v.sqrt())
    }

    #[inline]
    fn exp(&self) -> Self::Output {
        self.mapv(|v| v.exp())
    }
}
