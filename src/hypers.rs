use crate::types::Batch;

#[derive(Debug, Copy, Clone)]
pub struct Hypers {
	  pub learning_rate: f64,
    pub l2_rate: f64,
    pub batch_type: Batch,
}

impl Hypers {
    pub fn empty() -> Self{
        Self {
            learning_rate: 0.0,
            l2_rate: 0.0,
            batch_type: Batch::SGD,
        }
    }

    pub fn new(learning_rate: f64, l2_rate: f64) -> Self {
        Self {
            learning_rate,
            l2_rate,
            batch_type: Batch::SGD, // default
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
}
