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
    epsilon: f64, // ensures adam eq denominator is not zero, should sqrt of second moment be 0
}

impl Adam {
    // param shapes e.g. vec![(key, shape_value)]
    pub fn new() -> Self {
        Self {
            means: HashMap::new(),
            variances: HashMap::new(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    // smooth out the value using the past historical average, to get a blend of
    // the two values: average and most recent value
    fn smooth(decay_rate: f64, average: &Array2<f64>, value: &Array2<f64>) -> Array2<f64> {
        decay_rate*average + (1.0-decay_rate)*value
    }

    // offset the cost of initializing means to 0
    fn bias_correct(value: &Array2<f64>, decay_rate: f64, t: usize) -> Array2<f64> {
        value/(1.0-decay_rate.powf(t as f64 + 1.0))
    }
}

impl Optimizer for Adam {
    // param's key, value e.g. key: dw0, value: gradient
    fn calculate<'a>(&mut self, key: String, value: &'a Array2<f64>, t: usize) -> Cow<'a, Array2<f64>> {
        // momentum describes the fraction of the velocity that is ultimately applied to the learning rate
        if !self.means.contains_key(key.as_str()) && !self.variances.contains_key(key.as_str()) {
            let shape = (value.shape()[0], value.shape()[1]);
            self.means.insert(key.clone(), Array2::zeros(shape));
            self.variances.insert(key.clone(), Array2::zeros(shape));
        }

        let mean = self.means.get_mut(&key).unwrap();
        let variance = self.variances.get_mut(&key).unwrap();

        // update mean and variance values for param, e.g. dw or db
        *mean = Self::smooth(self.beta1, mean, value);
        *variance = Self::smooth(self.beta2, variance, &(value*value));

        let m_hat = Self::bias_correct(mean, self.beta1, t);
        let v_hat = Self::bias_correct(variance, self.beta2, t);

        // these are the adaptive momentum estimates
        // the numerator is the momentum term, the denominator is the rms prop term
        // momentum term to be applied to learning rate
        let momentum = m_hat/(v_hat.sqrt() + self.epsilon);

        Cow::Owned(momentum)
    }
}
