use ndarray::{s, Array1, Array2};
use std::collections::HashMap;

use crate::{
    activation::{ActFp, functions::Act}, cost::{CostDFp, CostCDFp}, layers::Batch,
    types::Classification, term_stack::{TT, TermStack}
};

#[derive(Debug)]
pub struct TermCache {
    pub stack: TermStack,
    one_hot: HashMap<usize, Array1<f64>>, // store list of one hot encoded vectors dim 1
    classification: Classification,
    learning_rate: f64,
    l2_rate: f64,
    batch_type: Batch,
    cost_d_fps: (CostDFp, CostCDFp),
    output_act_type: Act,
}

impl TermCache {
    pub fn new(
        backward: Vec<ActFp>, biases: &[Array2<f64>], output_size: usize,
        learning_rate: f64, l2_rate: f64, batch_type: Batch, cost_d_fps: (CostDFp, CostCDFp),
        output_act_type: Act
    ) -> Self {

        // Precompute one hot encoded vectors given output layer size
        let one_hot = Self::precompute(output_size);
        let classification = Classification::new(output_size);
        let stack = TermStack::new(backward, biases);

        TermCache {
            stack,
            one_hot,
            classification,
            learning_rate,
            l2_rate,
            batch_type,
            cost_d_fps,
            output_act_type,
        }
    }

    pub fn learning_rate(&self) -> f64 {
        match self.batch_type {
            Batch::SGD => self.learning_rate,
            Batch::Mini(batch_size) | Batch::Mini_(batch_size, _) =>
                self.learning_rate/batch_size as f64,
        }
    }

    pub fn l2_regularization_rate(&self) -> f64 { self.l2_rate }

    pub fn set_batch_type(&mut self, batch_type: Batch) {
        self.batch_type = batch_type;
    }

    // 1 One Hot Encoding

    // precompute one hot vector encodings for derivative calculations
    // given output layer size
    fn precompute(size: usize) -> HashMap<usize, Array1<f64>> {
        let mut map = HashMap::new();
        let mut zeros: Array1<f64>;

        // For example if the output size was two, the map would be:
        // 0 => [1,0] // top neuron fires if output 0
        // 1 => [0,1] // bottom neuron fires if output 1
        for i in 0..size {
            zeros = Array1::zeros(size);
            zeros[i] = 1.;
            map.insert(i, zeros);
        }

        map
    }

    fn one_hot(&self, index: usize) -> Option<&Array1<f64>> { self.one_hot.get(&index) }

    // 2 Activation & Cost Derivatives

    pub fn cost_derivative(&mut self, y: &Array2<f64>) -> Array2<f64> {
        let dc_da = self.partial_cost_derivative(y);
        let da_dz = self.nonlinear_derivative();

        // invoke cost specific combine deriv function with
        // output activation type
        (self.cost_d_fps.1)(dc_da, da_dz, self.output_act_type)
    }

    pub fn nonlinear_derivative(&mut self) -> Array2<f64> // returns da_dz
    {
        let z_last = self.stack.pop(TT::Linear).array();
        let a_derivative = self.stack.pop(TT::ActivationDerivative).fp();
        let da_dz = (a_derivative)(&z_last);
        da_dz
    }

    fn partial_cost_derivative(&mut self, y: &Array2<f64>) -> Array2<f64> { // returns dc_da
        let last_a: Array2<f64> = self.stack.pop(TT::Nonlinear).array();

        match self.classification {
            Classification::MultiClass(output_size) => {
                // Output labels is a matrix that accounts for output size and mini batch size

                // take the min if we are on the last batch and it doesn't contain
                // all batch size elements e.g. a remainder
                let actual_batch_size = std::cmp::min(self.batch_type.value(), y.shape()[0]);

                let mut output_labels: Array2<f64> =
                    Array2::zeros((output_size, actual_batch_size));

                // map y to output_labels by:
                // expanding each label value into a one hot encoded value - store result in normalized labels
                // perform for each label in batch

                // e.g. where y is 10 output size x 1 batch size or 10 output size x 32 batch size
                for i in 0..actual_batch_size {
                    let label = y[[i, 0]] as usize; // y is the label data, in form of a single column
                    // one hot encode the label, so 0 would be [1,0] and 1 would be [0,1] for output layer size 2
                    let encoded_label = self.one_hot(label).unwrap();
                    // assign encoded label on the column level (vertical)
                    output_labels.slice_mut(s![.., i]).assign(encoded_label); 
                }

                (self.cost_d_fps.0)(&last_a, &output_labels.view())
            },
          Classification::Binary => {
              // e.g. 1 x 1 or 1 x 32
              (self.cost_d_fps.0)(&last_a, &y.t())
          }
        }
    }
}

