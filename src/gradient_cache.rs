use ndarray::{s, Array1, Array2, CowArray, Ix2};
use std::collections::HashMap;

use crate::{
    activation::{Act, ActFp},
    cost::{CostDFp, CostCRFp},
    gradient_stack::{GradientStack},
    types::{Batch, Classification},
};

pub use crate::gradient_stack::GT;

#[derive(Debug)]
pub struct GradientCache {
    pub stack: GradientStack,
    one_hot: HashMap<usize, Array1<f64>>, // store list of one hot encoded vectors dim 1
    classification: Classification,
    batch_type: Batch,
    cost_derivative_fp: CostDFp,
    cost_combinate_fp: CostCRFp,
    output_act_type: Act,
}

impl GradientCache {
    pub fn new(
        backward: Vec<ActFp>,
        biases: &[Array2<f64>],
        output_size: usize,
        cost_d_fps: (CostDFp, CostCRFp),
        output_act_type: Act,
    ) -> Self {

        // Precompute one hot encoded vectors given output layer size
        let one_hot = Self::precompute(output_size);
        let classification = Classification::new(output_size);
        let stack = GradientStack::new(backward, biases);

        GradientCache {
            stack,
            one_hot,
            classification,
            batch_type: Batch::SGD,
            cost_derivative_fp: cost_d_fps.0,
            cost_combinate_fp: cost_d_fps.1,
            output_act_type,
        }
    }

    pub fn set_batch_type(&mut self, batch_type: Batch) {
        self.batch_type = batch_type;
    }

    pub fn add(&mut self, kind: GT, data: Array2<f64>) {
        match kind {
            GT::Linear => self.stack.push(kind, data),
            GT::Nonlinear => self.stack.push(kind, data),
            GT::IterationNew => self.stack.reset(data),
            _ => (),
        }
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

    fn one_hot(&self, index: usize) -> Option<&Array1<f64>> {
        self.one_hot.get(&index)
    }

    // 2 Activation & Cost Derivatives

    // returns dc_dz (not dc_da)
    pub fn cost_derivative(&mut self, y: &Array2<f64>) -> Array2<f64> {
        // cost derivative params
        let y_cow = self.one_hot_target(y);
        let last_a = self.stack.pop(GT::Nonlinear).array();

        // activation derivative params
        let last_z = self.stack.pop(GT::Linear).array();
        let a_derivative = *self.stack.pop(GT::ActivationDerivative).fp();

        // Generate appropriate combinate rule given particular cost function and activation type
        // Feed in relevant parameters

        let rule = (self.cost_combinate_fp)(self.cost_derivative_fp, last_a, y_cow.view(),
                                       a_derivative, last_z, self.output_act_type);

        rule.apply(y_cow.view())
    }

    pub fn activation_derivative(&mut self) -> Array2<f64> // returns da_dz
    {
        let last_z = self.stack.pop(GT::Linear).array();
        let a_derivative = self.stack.pop(GT::ActivationDerivative).fp();
        let da_dz = (a_derivative)(&last_z);

        da_dz
    }

    fn one_hot_target<'a, 'b>(&'a mut self, y: &'b Array2<f64>) -> CowArray<'b, f64, Ix2> {
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

                CowArray::from(output_labels)
            }
            Classification::Binary => {
                // e.g. 1 x 1 or 1 x 32
                CowArray::from(y.t())
            }
        }
    }
}
