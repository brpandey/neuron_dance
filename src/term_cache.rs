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
    output_size: usize,
    learning_rate: f64,
    batch_type: Batch,
    cost_d_fps: (CostDFp, CostCDFp),
    output_act_type: Act,
}

impl TermCache {
    pub fn new(
        backward: Vec<ActFp>, biases: &[Array2<f64>], output_size: usize,
        learning_rate: f64, batch_type: Batch, cost_d_fps: (CostDFp, CostCDFp),
        output_act_type: Act
    ) -> Self {

        // Precompute one hot encoded vectors given output layer size
        let one_hot = Self::precompute(output_size);
        let classification = if output_size > 1 { Classification::MultiClass } else { Classification::Binary };
        let stack = TermStack::new(backward, biases);

        TermCache {
            stack,
            one_hot,
            classification,
            output_size,
            learning_rate,
            batch_type,
            cost_d_fps,
            output_act_type,
        }
    }

    pub fn learning_rate(&self) -> f64 {
        match self.batch_type {
            Batch::SGD => self.learning_rate,
            Batch::Mini(batch_size) => self.learning_rate/batch_size as f64,
        }
    }

    pub fn set_batch_type(&mut self, batch_type: Batch) {
        self.batch_type = batch_type;
    }

    // 1 One Hot Encoding

    // precompute one hot vector encodings for derivative calculations
    // given output layer size
    fn precompute(output_size: usize) -> HashMap<usize, Array1<f64>> {
        let mut map = HashMap::new();
        let mut zeros: Array1<f64>;

        for i in 0..output_size {
            zeros = Array1::zeros(output_size);
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
        let da_dz = z_last.mapv(|v| a_derivative(v));
        da_dz
    }

    fn partial_cost_derivative(&mut self, y: &Array2<f64>) -> Array2<f64> { // returns dc_da
        let last_a: Array2<f64> = self.stack.pop(TT::Nonlinear).array();

        if let Classification::MultiClass = self.classification {
            // Output labels is a matrix that accounts for output size and mini batch size

            let mut output_labels: Array2<f64> =
                Array2::zeros((self.output_size, self.batch_type.value()));

            // map y to output_labels by:
            // expanding each label value into a one hot encoded value - store result in normalized labels
            // perform for each label in batch

            // e.g. where y is 10 x 1 or 10 x 32
            for i in 0..self.batch_type.value() { // 0..batch_size
                let label = y[[i, 0]] as usize;
                let encoded_label = self.one_hot(label).unwrap();
                output_labels.slice_mut(s![.., i]).assign(encoded_label);
            }

            (self.cost_d_fps.0)(&last_a, &output_labels.view()) //            &last_a - &output_labels
        } else {
            // e.g. 1 x 1 or 1 x 32
            (self.cost_d_fps.0)(&last_a, &y.t()) //            &last_a - &y.t()
        }
    }
}

