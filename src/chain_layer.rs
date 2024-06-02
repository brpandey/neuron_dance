/// Chain Layer

/// Compute Layer finds individual gradient or rate of change for each layer
use ndarray::{Array2, Axis};

use crate::chain_rule::ChainRule;

pub enum ComputeLayer {
    Output(OutputLayerTerms), // network's last layer of chain rule values
    Hidden(HiddenLayerTerms), // network's intermediate layer of chain rule values
}

// Output layer terms
// dc_db => dc_da * da_dz * dz_db   (or) dc_dz * dz_db
// dc_dw => dc_da * da_dz * dz_dw.t (or) dc_dz * dz_dw.t

pub struct OutputLayerTerms {
    pub dc_dz: Option<Array2<f64>>,
    pub dz_db: f64,
    pub dz_dw: Array2<f64>,
    pub bias_shape: (usize, usize),
}

// Hidden layer terms
// dc_db = dc_dz2 * dz2_da1 * da1_dz1 * dz1_db1   (or) dc_dz1 * dz1_db1
// dc_dw = dc_dz2 * dz2_da1 * da1_dz1 * dz1_dw1.t (or) dc_dz1 * dz1_dw1.t

pub struct HiddenLayerTerms {
    pub shared: SharedHiddenTerms,
    pub dz1_db1: f64,
    pub dz1_dw1: Array2<f64>,
    pub bias_shape: (usize, usize),
}
pub struct SharedHiddenTerms {
    pub dc_dz2: Array2<f64>,
    pub dz2_da1: Array2<f64>,
    pub da1_dz1: Array2<f64>,
    pub dc_dz1: Option<Array2<f64>>,
} // last field is result

impl ComputeLayer {
    // returns the acc in the chain rule computation aka dc_dz
    pub fn acc(&mut self) -> Option<Array2<f64>> {
        match self {
            ComputeLayer::Output(l2) => l2.dc_dz.take(),
            ComputeLayer::Hidden(l1) => l1.shared.dc_dz1.take(),
        }
    }
}

impl ChainRule for ComputeLayer {
    fn chain_rule(&mut self) -> (Array2<f64>, Array2<f64>) {
        match self {
            ComputeLayer::Output(l2) => l2.chain_rule(),
            ComputeLayer::Hidden(l1) => l1.chain_rule(),
        }
    }
}

impl ChainRule for OutputLayerTerms {
    fn chain_rule(&mut self) -> (Array2<f64>, Array2<f64>) {
        // Output layer terms
        // dc_db => dc_da * da_dz * dz_db   (or) dc_dz * dz_db
        // dc_dw => dc_da * da_dz * dz_dw.t (or) dc_dz * dz_dw.t

        // Handle batched computation correctly (only pertains to biases)
        // for example - if running with minibatches, suppose bias delta has 1x32 shape,
        // sum all the 32 (batch size) dc_db2 value contributions into one aggregate, making it 1x1 shape

        // Note: Z2 = W2A1 + B, dz_db is just the constant 1 that is multiplying B
        let dc_dz_ref = self.dc_dz.as_ref().unwrap();
        let dc_db_raw = (dc_dz_ref * self.dz_db).sum_axis(Axis(1)); // e.g. fold 1x32 into 1x1
        let dc_db = dc_db_raw.into_shape(self.bias_shape).unwrap(); // match bias shape in order to add/sub easily during update iteration

        // Z2 = W2A1 + B, dz_dw is just the constant A1 which is multiplying W2
        let dc_dw = dc_dz_ref.dot(&self.dz_dw.t());

        (dc_db, dc_dw)
    }
}

impl ChainRule for HiddenLayerTerms {
    fn chain_rule(&mut self) -> (Array2<f64>, Array2<f64>) {
        // Main equations - hidden layer terms
        // dc_db = dc_dz2 * dz2_da1 * da1_dz1 * dz1_db1   (or) dc_dz1 * dz1_db1
        // dc_dw = dc_dz2 * dz2_da1 * da1_dz1 * dz1_dw1.t (or) dc_dz1 * dz1_dw1.t

        let dc_da1 = self.shared.dz2_da1.t().dot(&self.shared.dc_dz2);
        let dc_dz1 = dc_da1 * &self.shared.da1_dz1;

        // handle batched computation correctly (only pertains to biases)
        // for example - if running with minibatches, suppose bias delta has 3x32 shape,
        // sum all the 32 (batch size) dc_db1 value contributions into one aggregate, making it 3x1 shape
        let dc_db1_raw = (&dc_dz1 * self.dz1_db1).sum_axis(Axis(1));
        let dc_db1 = dc_db1_raw.into_shape(self.bias_shape).unwrap(); // cost derivative with respect to bias

        let dc_dw1 = dc_dz1.dot(&self.dz1_dw1.t()); // cost derivative with respect to weight

        self.shared.dc_dz1 = Some(dc_dz1); // return acc

        (dc_db1, dc_dw1)
    }
}
