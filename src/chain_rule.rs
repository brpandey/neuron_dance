use std::collections::VecDeque;
use ndarray::{Array2, Axis};
use crate::term_cache::{TermCache, TT};

#[derive(Debug)]
pub struct ChainRuleComputation<'a> {
    pub tc: &'a mut TermCache,
    pub bias_deltas: VecDeque<Array2<f64>>,
    pub weight_deltas: VecDeque<Array2<f64>>,
}

impl <'a> ChainRuleComputation<'a> {
    pub fn new(tc: &'a mut TermCache) -> Self {
        ChainRuleComputation {
            tc,
            weight_deltas: VecDeque::new(),
            bias_deltas: VecDeque::new(),
        }
    }

    pub fn bias_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> { // iterator is tied to the lifetime of current computation
        self.bias_deltas.iter()
    }

    pub fn weight_deltas(&self) -> impl Iterator<Item = &'_ Array2<f64>> { // iterator is tied to the lifetime of current computation
        self.weight_deltas.iter()
    }

    // Compute chain rule for last/output layer
    // For example purposes consider a 3 layer NN including an input layer
    // with these equations:

    /*
       C  = (A2 - Y)^2
       A2 = sigmoid(Z2)
       Z2 = W2*A1 + B2
       A1 = relu(Z1)
       Z1 = W1*X + B1

       Where * is the dot product of two matrices l*m and m*n resulting in a matrix l*n
    */

    pub fn init(&mut self, y: &Array2<f64>) -> Array2<f64> {
        // Main chain rule equations
        // dc_db => dc_da * da_dz * dz_db   (or) dc_dz * dz_db
        // dc_dw => dc_da * da_dz * dz_dw.t (or) dc_dz * dz_dw.t

        // create shared component
        let shared = SharedOutputTerms {
            dc_da: self.tc.cost_derivative(y), // e.g. C = (A2 − Y)^2
            da_dz: self.tc.nonlinear_derivative(),  // e.g A2 = sigmoid (Z2)
            dc_dz: None,
        };

        // create current layer's terms
        let mut layer_terms = Layer::Output(
            OutputLayerTerms {
                shared,
                dz_db: 1.0,
                dz_dw: self.tc.stack.pop(TT::Nonlinear).array(),
                bias_shape: self.tc.stack.pop(TT::BiasShape).shape(),
            }
        );

        let results = layer_terms.chain_rule();

        self.bias_deltas.push_front(results.0); // fold result into deque
        self.weight_deltas.push_front(results.1); // fold result into deque

        layer_terms.acc().unwrap()
    }

    // Method can be called repeatedly:
    // Computes chain rule from preceding layer value (layer j) and calculates for current layer (layer i)
    // With layer j being layer after layer i in a feed forward nn, so e.g. j is 2 and i 1
    // Bias and weight delta values are folded into the respective collections
    pub fn fold_layer(&mut self, dc_dz2: Array2<f64>, w: &Array2<f64>) -> Array2<f64>{
        // Main equations
        // dc_db = dc_dz2 * dz2_da1 * da1_dz1 * dz1_db1   (or) dc_dz1 * dz1_db1
        // dc_dw = dc_dz2 * dz2_da1 * da1_dz1 * dz1_dw1.t (or) dc_dz1 * dz1_dw1.t

        // create shared component
        let shared = SharedHiddenTerms {
            dc_dz2,
            dz2_da1: w.clone(), // Z2 = W2A1 + B, w is just W2
            da1_dz1: self.tc.nonlinear_derivative(), // derivative of e.g. relu applied to Z1,
            dc_dz1: None
        }; // last field is result

        // current layer's terms
        let mut layer_terms = Layer::Hidden(
            HiddenLayerTerms {
                shared,
                dz1_db1: 1.0,         // For example Z1 = W1X + B1
                dz1_dw1: self.tc.stack.pop(TT::Nonlinear).array(),
                bias_shape: self.tc.stack.pop(TT::BiasShape).shape(),
            }
        );

        let results = layer_terms.chain_rule();

        self.bias_deltas.push_front(results.0); // fold result into deque
        self.weight_deltas.push_front(results.1); // fold result into deque

        layer_terms.acc().unwrap()  // return acc
    }
}

enum Layer {
    Output(OutputLayerTerms), // network's last layer of chain rule values
    Hidden(HiddenLayerTerms), // network's intermediate layer of chain rule values
}

// Output layer terms
// dc_db => dc_da * da_dz * dz_db   (or) dc_dz * dz_db
// dc_dw => dc_da * da_dz * dz_dw.t (or) dc_dz * dz_dw.t

struct OutputLayerTerms { shared: SharedOutputTerms, dz_db: f64, dz_dw: Array2<f64>, bias_shape: (usize, usize)}
struct SharedOutputTerms { dc_da: Array2<f64>, da_dz: Array2<f64>, dc_dz: Option<Array2<f64>> } // last field is result

// Hidden layer terms
// dc_db = dc_dz2 * dz2_da1 * da1_dz1 * dz1_db1   (or) dc_dz1 * dz1_db1
// dc_dw = dc_dz2 * dz2_da1 * da1_dz1 * dz1_dw1.t (or) dc_dz1 * dz1_dw1.t

struct HiddenLayerTerms { shared: SharedHiddenTerms, dz1_db1: f64, dz1_dw1: Array2<f64>, bias_shape: (usize, usize)}
struct SharedHiddenTerms { dc_dz2: Array2<f64>, dz2_da1: Array2<f64>, da1_dz1: Array2<f64>, dc_dz1: Option<Array2<f64>>} // last field is result

impl Layer {
    fn acc(&mut self) -> Option<Array2<f64>> {
        match self {
            Layer::Output(l2) => return l2.shared.dc_dz.take(),
            Layer::Hidden(l1) => return l1.shared.dc_dz1.take()
        }
    }
}

pub trait ChainRule {
    fn chain_rule(&mut self) -> (Array2<f64>, Array2<f64>);
}


impl ChainRule for Layer {
    fn chain_rule(&mut self) -> (Array2<f64>, Array2<f64>) {
        match self {
            Layer::Output(l2) => l2.chain_rule(),
            Layer::Hidden(l1) => l1.chain_rule(),
        }
    }
}


impl ChainRule for OutputLayerTerms {
    fn chain_rule(&mut self) -> (Array2<f64>, Array2<f64>) {
        // Output layer terms
        // dc_db => dc_da * da_dz * dz_db   (or) dc_dz * dz_db
        // dc_dw => dc_da * da_dz * dz_dw.t (or) dc_dz * dz_dw.t

        let dc_dz: Array2<f64> = &self.shared.dc_da * &self.shared.da_dz;

        // Handle batched computation correctly (only pertains to biases)
        // for example - if running with minibatches, suppose bias delta has 1x32 shape,
        // sum all the 32 (batch size) dc_db2 value contributions into one aggregate, making it 1x1 shape

        // Note: Z2 = W2A1 + B, dz_db is just the constant 1 that is multiplying B

        let dc_db_raw = (&dc_dz * self.dz_db).sum_axis(Axis(1)); // e.g. fold 1x32 into 1x1
        let dc_db = dc_db_raw.into_shape(self.bias_shape).unwrap(); // match bias shape in order to add/sub easily during update iteration

        // Z2 = W2A1 + B, dz_dw is just the constant A1 which is multiplying W2
        let dc_dw = dc_dz.dot(&self.dz_dw.t());

        self.shared.dc_dz = Some(dc_dz);

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
