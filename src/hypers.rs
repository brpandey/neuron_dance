use std::default::Default;
use nanoserde::{DeBin, SerBin}; 
use crate::types::Batch;
use crate::optimizer::{Optim, Optimizer};
use crate::activation::Act;
use crate::cost::Loss;
use crate::save::Save;
use crate::archive::Archive;

#[derive(Clone, Debug, Default, DeBin, SerBin, PartialEq)]
pub struct Hypers {
    learning_rate: f64,
    l2_rate: f64,
    input_size: usize,
    class_size: usize,
    activations: Vec<Act>,
    loss_type: Loss,
    batch_type: Batch,
    optimizer_type: Optim,
}

impl Hypers {
    pub fn new(learning_rate: f64, l2_rate: f64, input_size: usize, class_size: usize, activations: Vec<Act>, loss_type: Loss) -> Self {
        Self {
            learning_rate,
            l2_rate,
            input_size,
            class_size,
            activations,
            loss_type,
            ..Default::default()
        }
    }

    pub fn learning_rate(&self) -> f64 {
        match self.batch_type {
            Batch::SGD => self.learning_rate,
            Batch::Mini(batch_size) | Batch::Mini_(batch_size, _) => {
                self.learning_rate / batch_size as f64
            }
        }
    }

    pub fn input_size(&self) -> usize { self.input_size }

    pub fn l2_regularization_rate(&self) -> f64 {
        self.l2_rate
    }

    pub fn set_batch_type(&mut self, batch_type: Batch) {
        self.batch_type = batch_type;
    }

    pub fn set_optimizer(&mut self, o_type: Optim) {
        self.optimizer_type = o_type;
    }

    pub fn optimizer(&mut self) -> Box<dyn Optimizer> {
        self.optimizer_type.into()
    }

    pub fn optimizer_type(&self) -> Optim {
        self.optimizer_type
    }

    pub fn loss_type(&self) -> Loss { self.loss_type }
    pub fn activations(&self) -> &'_ Vec<Act> { self.activations.as_ref() }
    pub fn class_size(&self) -> usize { self.class_size }
}

impl Archive for Hypers {}

// Hypers doesn't require a new intermediate proxy structure hence use self
impl Save for Hypers {
    type Proxy = Hypers;
}

impl From<&Hypers> for Hypers {
    fn from(h: &Hypers) -> Self { h.clone() }
}
