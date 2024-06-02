/// Nadam - Nesterov-accelerated Adaptive Moment Estimation

/// Nadam extends the Adam optimization algorithm by using Nesterov Momentum
/// in its bias correction for hyper mu
use ndarray::Array2;
use std::borrow::Cow;
use std::collections::HashMap;

use crate::algebra::AlgebraExt;
use crate::optimizer::{adam::Adam, CompositeKey, HistType, Optimizer, ParamKey};

pub struct NAdam {
    hist_types: Vec<HistType>,
    historical: HashMap<CompositeKey, Array2<f64>>,
    mu: f64,      // first order moment mu
    nu: f64,      // second order moment nu
    epsilon: f64, // stabilizer to ensure denominator is never zero, should sqrt of second moment be 0
}

impl NAdam {
    pub fn new() -> Self {
        Self {
            hist_types: Vec::from(&[HistType::Mean, HistType::Variance]),
            historical: HashMap::new(),
            mu: 0.975,
            nu: 0.999,
            epsilon: 1e-8,
        }
    }

    // bias correct mu
    fn bias_correct1(value: &Array2<f64>, decay_rate: f64, gradient: &Array2<f64>) -> Array2<f64> {
        (decay_rate * value / (1.0 - decay_rate))
            + ((1.0 - decay_rate) * gradient / (1.0 - decay_rate))
    }

    // bias correct nu
    fn bias_correct2(value: &Array2<f64>, decay_rate: f64) -> Array2<f64> {
        (decay_rate * value) / (1.0 - decay_rate)
    }
}

impl Optimizer for NAdam {
    // Produce adjustment value given specific param's key and value e.g. "dw0", and gradient value dw
    fn calculate<'a>(
        &mut self,
        key: ParamKey,
        value: &'a Array2<f64>,
        _t: usize,
    ) -> Cow<'a, Array2<f64>> {
        Adam::initialize(key, value.raw_dim(), &self.hist_types, &mut self.historical);

        let mean = self
            .historical
            .get_mut(&CompositeKey(key, HistType::Mean))
            .unwrap();
        *mean = mean.smooth(self.mu, value, false);
        let m_hat = Self::bias_correct1(mean, self.mu, value);

        let variance = self
            .historical
            .get_mut(&CompositeKey(key, HistType::Variance))
            .unwrap();
        *variance = variance.smooth(self.nu, value, true);
        let v_hat = Self::bias_correct2(variance, self.nu);
        let momentum = m_hat / (v_hat.sqrt() + self.epsilon);

        Cow::Owned(momentum)
    }
}
