use std::default::Default;
use crate::types::Batch;
use crate::optimizer::{Optim, Optimizer};
use crate::activation::Act;
use crate::cost::Loss;

#[derive(Debug, Default)]
pub struct Hypers {
    pub learning_rate: f64,
    pub l2_rate: f64,
    pub optimizer: Option<Box<dyn Optimizer>>,
    pub class_size: usize,
    pub activations: Vec<Act>,
    pub loss_type: Loss,
    pub batch_type: Batch,
    pub optimizer_type: Optim,
}

impl Hypers {
    pub fn new(learning_rate: f64, l2_rate: f64, class_size: usize, activations: Vec<Act>, loss_type: Loss) -> Self {
        let optimizer = Some(Optim::Default.into());

        Self {
            learning_rate,
            l2_rate,
            optimizer,
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

    pub fn l2_regularization_rate(&self) -> f64 {
        self.l2_rate
    }

    pub fn set_batch_type(&mut self, batch_type: Batch) {
        self.batch_type = batch_type;
    }

    pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>, o_type: Optim) {
        self.optimizer = Some(optimizer);
        self.optimizer_type = o_type;
    }

    pub fn optimizer(&mut self) -> &mut Box<dyn Optimizer> {
        self.optimizer.as_mut().unwrap()
    }

    pub fn optimizer_type(&self) -> Optim {
        self.optimizer_type
    }

    pub fn loss_type(&self) -> Loss { self.loss_type }

    pub fn activations(&self) -> &'_ [Act] { self.activations.as_ref() }

    pub fn output_classes(&self) -> usize { self.class_size }
}
