/// AMSGrad

/// AMSGrad is also an extension to Adam with the addition of a max variance second moment variable
/// to help bias correct the variance, which prevents a rapid optimization deceleration

use ndarray::Array2;
use std::collections::HashMap;
use std::borrow::Cow;

use crate::optimizer::{Optimizer, CompositeKey, ParamKey, HistType, adam::Adam};
use crate::algebra::AlgebraExt;

pub struct AmsGrad {
    hist_types: Vec<HistType>,
    historical: HashMap<CompositeKey, Array2<f64>>,
    beta1: f64, // first order moment beta
    beta2: f64, // second order moment beta
    epsilon: f64, // stabilizer to ensure denominator is never zero, should sqrt of second moment be 0
}

impl AmsGrad {
    pub fn new() -> Self {
        Self {
            hist_types: Vec::from(&[HistType::Mean, HistType::Variance, HistType::Vhat]),
            historical: HashMap::new(),
            beta1: 0.9, // use 90 percent of prior historical average, but only 10% of new value
            beta2: 0.999, // use 99 percent of prior historical average -- avoid fluctuations!
            epsilon: 1e-8,
        }
    }
}

impl Optimizer for AmsGrad {
    // Produce adjustment value given specific param's key and value e.g. "dw0", and gradient value dw
    fn calculate<'a>(&mut self, key: ParamKey, value: &'a Array2<f64>, t: usize) -> Cow<'a, Array2<f64>> {
        Adam::initialize(key, value.raw_dim(), &self.hist_types, &mut self.historical);

        let m_key = CompositeKey(key, HistType::Mean);
        let v_key = CompositeKey(key, HistType::Variance);
        let vh_key = CompositeKey(key, HistType::Vhat);

        let mean = self.historical.get(&m_key).unwrap();
        let variance = self.historical.get(&v_key).unwrap();

        let m = mean.smooth(self.beta1.powf(t as f64 + 1.0), value, false);
        let v = variance.smooth(self.beta2, value, true);

        let v_hat = self.historical.get(&vh_key).unwrap();
        let vh = v_hat.maximum(&v); // take element-wise maximum of current and preceding v_hat

        let momentum = &m/(vh.sqrt() + self.epsilon);

        // update
        self.historical.insert(m_key, m);
        self.historical.insert(v_key, v);
        self.historical.insert(vh_key, vh);

        Cow::Owned(momentum)
    }
}

