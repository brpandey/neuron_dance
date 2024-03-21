use either::*;
use std::{ops::RangeBounds, convert::Into, fmt, fmt::Debug};

use crate::activation::{Activation, ActFp};

// Make fields public for layers literal construction

pub struct Input1(pub usize);
pub struct Input2(pub usize, pub usize);
pub struct Dense(pub usize, pub Act);

// re-export types into Layer
pub use crate::activation::functions::Act;
pub use crate::cost::functions::Loss;
pub use crate::network::{Batch, Eval};
pub use crate::metrics::{Mett};

pub trait Layer {
    type Output;
    fn reduce(&self) -> Self::Output;
}

#[derive(PartialEq, PartialOrd)]
pub enum LayerOrder {
    OnlyFirst,
    OnlySecond,
    FromSecondToLast,
}

impl LayerOrder {
    fn value(&self) -> Either<usize, impl RangeBounds<usize>> {
        match self {
            LayerOrder::OnlyFirst => Left(0),
            LayerOrder::OnlySecond => Left(1),
            LayerOrder::FromSecondToLast => Right(1..),
        }
    }

    fn valid(&self, layer_index: usize) -> bool {
        match self.value() {
            Left(idx) => idx == layer_index,
            Right(ref range) => range.contains(&layer_index)
        }
    }
}

pub enum LayerTerms {
    Input(LayerOrder, usize), // order, input size
    Dense(LayerOrder, usize, Box<dyn Activation>), // order, dense siz, act
}

impl Layer for Input1 {
    type Output = LayerTerms;

    fn reduce(&self) -> Self::Output {
        LayerTerms::Input(LayerOrder::OnlyFirst, self.0)
    }
}

impl Layer for Input2 {
    type Output = LayerTerms;

    fn reduce(&self) -> Self::Output {
        LayerTerms::Input(LayerOrder::OnlyFirst, self.0 * self.1)
    }
}

impl Layer for Dense {
    type Output = LayerTerms;

    fn reduce(&self) -> Self::Output {
        let size = self.0;
        let act: Box<dyn Activation> = self.1.into();
        LayerTerms::Dense(LayerOrder::FromSecondToLast, size, act)
    }
}

pub struct LayerStack(Vec<Box<dyn Layer<Output = LayerTerms>>>); // holds a vec of layer trait objects

impl Debug for LayerStack {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "LayerStack contains {:?} layers", &self.len())
    }
}

impl LayerStack {
    pub fn new() -> Self {
        LayerStack(vec![])
    }

    pub fn add<L: Layer<Output = LayerTerms> + 'static>(&mut self, l: L) {
        self.0.push(Box::new(l));
    }

    pub fn len(&self) -> usize { self.0.len() }
}

impl Layer for LayerStack {
    //tuple of vecs: (sizes, act fps, act deriv fps)
    type Output = (Vec<usize>, Vec<ActFp>, Vec<ActFp>);

    fn reduce(&self) -> Self::Output {
        let acc = (vec![0], vec![], vec![]); // acc is type Output
        self.0.iter().enumerate().fold(acc, |mut acc, (i,l)| {
            match l.reduce() {
                LayerTerms::Input(ref order, size) if order.valid(i) => {
                    acc.0.push(size); // input size
                    // swap_remove dummy value 0 with last element -> input size
                    acc.0.swap_remove(0);
                    acc
                },
                LayerTerms::Dense(ref order, size, act) if order.valid(i) => {
                    let (act_fp, deriv_fp) = act.pair(); // activation and activation derivative functions
                    acc.0.push(size);
                    acc.1.push(act_fp);
                    acc.2.push(deriv_fp);
                    acc
                },
                _ => panic!("Layer order is incorrect, perhaps input layer is not first layer added?"),
            }
        })
    }
}
