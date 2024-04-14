/// Algebra

/// AlgebraExt provides extended tensor convenience functions specifically for Array2
/// Could be used to represent n dimensional tensors at some point

/// Array2<f64> is ArrayBase<OwnedRepr<f64>, Ix2>
/// ArrayView2<f64> is ArrayBase<ViewRepr<&'a f64>, Ix2>>;

use ndarray::{Array1, Array2, Axis, Zip};

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
    fn maximum(&self, other: &Self) -> Self::Output;
    fn sqrt(&self) -> Self::Output;
    fn smooth(&self, decay_rate: f64, value: &Array2<f64>) -> Self::Output;
}

impl AlgebraExt for Array2<f64> {
    type Output = Self;
    type Output1 = Array1<f64>;

    // finds tensor index containing highest degree of belief (highest activation)
    fn arg_max(&self) -> usize {
        let mut max_acc_index = 0;

        // if we have a single neuron output, return either 0 or 1
        if self.shape() == &[1,1] {
            return self[[0, 0]].round() as usize
        }

        // start by comparing value from first column (0) of 2d array by index
        for (i, &v) in self.iter().enumerate() {
            if v > self[[max_acc_index, 0]] { 
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

    // Smooth out the value using the past historical average (self), to get a blend of
    // the two values: average and most recent value (e.g. specific param's gradient)
    // greater weight is given to the historical average and less weight to the new value

    #[inline]
    fn smooth(&self, decay_rate: f64, value: &Array2<f64>) -> Self::Output { 
        decay_rate*self + (1.0-decay_rate)*value
    }

    fn maximum(&self, other: &Self) -> Self::Output {
        let mut output = Array2::<f64>::zeros(self.raw_dim());

        if self.shape() == other.shape() {
            Zip::from(&mut output)
                .and(self)
                .and(other)
                .for_each(|m, &x, &y| *m = x.max(y));
        }

        output
    }
}
