/// Algebra

/// AlgebraExt provides extended tensor convenience functions specifically for Array2
/// Could be used to represent n dimensional tensors at some point

/// Array2<f64> is ArrayBase<OwnedRepr<f64>, Ix2>
/// ArrayView2<f64> is ArrayBase<ViewRepr<&'a f64>, Ix2>>;
use ndarray::{Array1, Array2, Axis, Zip};
use ndarray_stats::QuantileExt;

#[allow(dead_code)]
pub trait AlgebraExt<W = Self, B = Self> {
    type Output;
    type Output1;

    fn arg_max(&self) -> usize;
    fn weighted_sum(&self, w: &W, b: &B) -> Self::Output;
    fn normalize(&self) -> f64;
    fn min(&self) -> f64;
    fn max(&self) -> f64;
    fn min_axis(&self, axis: Axis) -> Self::Output1;
    fn max_axis(&self, axis: Axis) -> Self::Output1;
    fn maximum(&self, other: &Self) -> Self::Output;
    fn smooth(&self, decay_rate: f64, value: &Array2<f64>, square_value: bool) -> Self::Output;
}

impl AlgebraExt for Array2<f64> {
    type Output = Self;
    type Output1 = Array1<f64>;

    // finds tensor index containing highest degree of belief (highest activation)
    fn arg_max(&self) -> usize {
        let mut max_acc_index = 0;

        // if we have a single neuron output, return either 0 or 1
        if self.shape() == &[1, 1] {
            return self[[0, 0]].round() as usize;
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
    fn normalize(&self) -> f64 {
        (self * self).sum().sqrt()
    }

    fn min(&self) -> f64 {
        *QuantileExt::min(self).unwrap()
    }

    #[inline]
    fn min_axis(&self, axis: Axis) -> Self::Output1 {
        self.map_axis(axis, |v| *v.iter().min_by(|a, b| a.total_cmp(b)).unwrap())
    }

    fn max(&self) -> f64 {
        *QuantileExt::max(self).unwrap()
    }

    #[inline]
    fn max_axis(&self, axis: Axis) -> Self::Output1 {
        self.map_axis(axis, |v| *v.iter().max_by(|a, b| a.total_cmp(b)).unwrap())
    }

    // Smooth out the value using the past historical average (self), to get a blend of
    // the two values: average and most recent value (e.g. specific param's gradient)
    // greater weight is given to the historical average and less weight to the new value

    fn smooth(&self, decay_rate: f64, value: &Array2<f64>, square_value: bool) -> Self::Output {
        let mut output = Array2::<f64>::zeros(self.raw_dim());

        Zip::from(&mut output)
            .and(self)
            .and(value)
            .for_each(|o, &s, &v| {
                if square_value {
                    *o = decay_rate * s + (1.0 - decay_rate) * v * v
                } else {
                    *o = decay_rate * s + (1.0 - decay_rate) * v
                }
            });

        output
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
