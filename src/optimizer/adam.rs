/// Adam - Adaptive Moment Estimation (uses an adaptive learning rate for each param)

/// Gradient descent optimization uses a constant learning rate
/// to take a fraction of the gradient to update a given model parameter (e.g. w or b)

/// Instead of relying on a constant fraction of the gradient, Adam makes it adaptive,
/// by modifying the fraction of the gradient for each parameter and each revision
/// by leveraging more information specifically the current and historical gradients

/// The variance term (from RMSProp), uses the squares of the mean to avoid having a term that can
/// be negative, and uses square root to closely match the gradient as to not over shoot the
/// rate at which the gradient changes [RMS = root (square root), mean, square]

/// The mean or velocity term borrows velocity from preceding revisions.  This is useful
/// especially as we reach the trough of our loss function when the gradient contributions get smaller and smaller
/// and the learning rate is still constant. It prevents the velocity from slowing down by slightly boosting it

use ndarray::Array2;
use std::collections::HashMap;
use std::borrow::Cow;

use crate::optimizer::Optimizer;
use crate::algebra::AlgebraExt;

pub struct Adam {
    means: HashMap<String, Array2<f64>>, // first order mean value for param value
    variances: HashMap<String, Array2<f64>>, // second order variances for param value
    beta1: f64, // first order moment beta
    beta2: f64, // second order moment beta
    epsilon: f64, // stabilizer to ensure denominator is never zero, should sqrt of second moment be 0
}

impl Adam {
    pub fn new() -> Self {
        Self {
            means: HashMap::new(),
            variances: HashMap::new(),
            beta1: 0.9, // use 90 percent of prior historical average, but only 10% of new value
            beta2: 0.999, // use 99 percent of prior historical average -- avoid fluctuations!
            epsilon: 1e-8,
        }
    }

    // Smooth out the value using the past historical average, to get a blend of
    // the two values: average and most recent value (e.g. specific param's gradient)
    // greater weight is given to the historical average and less weight to the new value
    fn smooth(decay_rate: f64, average: &Array2<f64>, value: &Array2<f64>) -> Array2<f64> {
        decay_rate*average + (1.0-decay_rate)*value
    }

    // Offset the cost of originally initializing value to 0
    fn bias_correct(value: &Array2<f64>, decay_rate: f64, t: usize) -> Array2<f64> {
        value/(1.0-decay_rate.powf(t as f64 + 1.0))
    }
}

impl Optimizer for Adam {
    // Produce adjustment value given specific param's key and value e.g. "dw0", and gradient value dw
    fn calculate<'a>(&mut self, key: String, value: &'a Array2<f64>, t: usize) -> Cow<'a, Array2<f64>> {

        if !self.means.contains_key(key.as_str()) && !self.variances.contains_key(key.as_str()) {
            let shape = (value.shape()[0], value.shape()[1]);
            self.means.insert(key.clone(), Array2::zeros(shape));
            self.variances.insert(key.clone(), Array2::zeros(shape));
        }

        let mean = self.means.get_mut(&key).unwrap();
        let variance = self.variances.get_mut(&key).unwrap();

        // Smooth two historical averages blending in new value so as not to be
        // susceptible to gradient variations, then update values for each param, e.g. dw or db
        *mean = Self::smooth(self.beta1, mean, value);
        *variance = Self::smooth(self.beta2, variance, &(value*value));

        // Account for errors/biases since mean and variance were initalized to 0's
        let m_hat = Self::bias_correct(mean, self.beta1, t);
        let v_hat = Self::bias_correct(variance, self.beta2, t);

        // The adaptive momentum estimates are two fold:
        // the numerator is the momentum term, the denominator is the rms prop term

        // if the variance term is large, meaning the gradients have a larger spread =>
        // the rate of change should reduce more slowly than the rate at which the gradient reduces
        // which helps to avoid big fluctuations, if the variance term is small, it means a larger rate of change
        // since inversely proportional => 1/G

        // if the historical mean term is large => don't slow down the rate of change
        // keep the past "momentum" going
        let momentum = m_hat/(v_hat.sqrt() + self.epsilon);

        // momentum describes the fraction of the velocity that is ultimately applied to the learning rate
        Cow::Owned(momentum)
    }
}
