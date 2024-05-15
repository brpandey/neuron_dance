use std::default::Default;
use nanoserde::{DeBin, SerBin}; // tiny footprint and fast!
use crate::types::Batch;
use crate::optimizer::{Optim, Optimizer};
use crate::activation::Act;
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

use crate::activation::ActivationStrings;

#[derive(Clone, Debug, Default, DeBin, SerBin)]
pub struct HypersArchive { // subset of Hypers
    pub learning_rate: f64,
    pub l2_rate: f64,
    pub class_size: usize,
    pub activations: ActivationStrings, //vec of activation strings
    pub loss_type: String, // loss as string type
    pub batch_type: String, // batch as string type
    pub optimizer_type: String, // optimizer as String type
}

impl Archive for HypersArchive {}

impl Save for Hypers {
    type Target = HypersArchive;

    fn to_archive(&self) -> Self::Target {
        dbg!(&self.batch_type);
        dbg!(&self.batch_type.to_string());

        HypersArchive {
            learning_rate: self.learning_rate,
            l2_rate: self.l2_rate,
            class_size: self.class_size,
            activations: (&self.activations).into(),
            loss_type: self.loss_type.to_string(),
            batch_type: self.batch_type.to_string(),
            optimizer_type: self.optimizer_type.to_string(),
        }
    }

    fn from_archive(archive: Self::Target) -> Self {
        let opt: Box<dyn Optimizer> =
            archive.optimizer_type.parse::<Optim>().unwrap().into();

        let acts: Vec<Act> = archive.activations.into();

        dbg!(&archive.batch_type);
        dbg!(&archive.batch_type.parse::<Batch>());

        Hypers {
            learning_rate: archive.learning_rate,
            l2_rate: archive.l2_rate,
            optimizer: Some(opt),
            class_size: archive.class_size,
            activations: acts,
            loss_type: archive.loss_type.parse().unwrap(),
            batch_type: archive.batch_type.parse().unwrap(),
            optimizer_type: archive.optimizer_type.parse().unwrap(),
        }
    }
}
