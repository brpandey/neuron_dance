use std::default::Default;
use nanoserde::{DeBin, SerBin}; // tiny footprint and fast!
use crate::types::{Batch, SimpleError};
use crate::optimizer::{Optim, Optimizer};
use crate::activation::{Act, ActivationStrings};
use crate::cost::Loss;
use crate::save::{Save, Archive};

#[derive(Debug, Default)]
pub struct Hypers {
    learning_rate: f64,
    l2_rate: f64,
    optimizer: Option<Box<dyn Optimizer>>,
    class_size: usize,
    activations: Vec<Act>,
    loss_type: Loss,
    batch_type: Batch,
    optimizer_type: Optim,
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
    pub fn activations(&self) -> &'_ Vec<Act> { self.activations.as_ref() }
    pub fn class_size(&self) -> usize { self.class_size }
}

#[derive(Clone, Debug, Default, DeBin, SerBin)]
pub struct HypersArchive(f64, f64, usize, ActivationStrings, String, String, String);
// above fields: 0 learning_rate, 1 l2_rate, 3 class_size, 4 activations, 5 loss_type, 6 batch_type 7 optimizer_type

impl Archive for HypersArchive {}

impl Save for Hypers {
    type Target = HypersArchive;

    fn to_archive(&self) -> Self::Target { self.into() }
    fn from_archive(archive: Self::Target) -> Result<Self, SimpleError> { Ok(Hypers::from(archive)) }
}

impl From<HypersArchive> for Hypers {
    fn from(ar: HypersArchive) -> Hypers {
        Hypers {
            learning_rate: ar.0, l2_rate: ar.1,
            optimizer: Some(ar.6.parse::<Optim>().unwrap().into()),
            class_size: ar.2, activations: ar.3.into(),
            loss_type: ar.4.parse().unwrap(), batch_type: ar.5.parse().unwrap(),
            optimizer_type: ar.6.parse().unwrap(),
        }
    }
}

impl From<&Hypers> for HypersArchive {
    fn from(h: &Hypers) -> HypersArchive {
        HypersArchive(
            h.learning_rate, h.l2_rate, h.class_size,
            (&h.activations).into(), h.loss_type.to_string(),
            h.batch_type.to_string(), h.optimizer_type.to_string()
        )
    }
}

