use crate::types::Batch;
use crate::optimizer::{Optim, Optimizer};

#[derive(Debug)]
pub struct Hypers {
	  pub learning_rate: f64,
    pub l2_rate: f64,
    pub batch_type: Batch,
    pub optimizer: Option<Box<dyn Optimizer>>,
    pub optimizer_type: Optim,
}

impl Hypers {
    pub fn empty() -> Self{
        Self {
            learning_rate: 0.0,
            l2_rate: 0.0,
            batch_type: Batch::SGD,
            optimizer: None,
            optimizer_type: Optim::Default,
        }
    }

    pub fn new(learning_rate: f64, l2_rate: f64) -> Self {
        Self {
            learning_rate,
            l2_rate,
            batch_type: Batch::SGD, // default
            optimizer: None,
            optimizer_type: Optim::Default,
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
}
