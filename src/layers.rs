use either::*;
use std::{ops::RangeBounds, convert::Into, fmt, fmt::Debug};

use crate::activation::{Activation, ActFp};

// Make fields public for layers literal construction

pub struct Input1(pub usize);
pub struct Input2(pub usize, pub usize);
pub struct Dense(pub usize, pub Act);

// re-export types into Layer, to consolidate interface
pub use crate::activation::Act;
pub use crate::cost::Loss;
pub use crate::types::{Batch, Eval, Metr};
pub use crate::weight::Weit;
pub use crate::optimizer::Optim;

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
    Dense(LayerOrder, usize, Box<dyn Activation>, Act, Weit), // order, dense siz, act
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
        let act_type = self.1;
        let act: Box<dyn Activation> = self.1.into();

        let w_distr = match act_type {
            // act functions w default weight initializations
            Act::Tanh | Act::Sigmoid => Weit::GlorotU,
            Act::Relu => Weit::He,
            // act functions with weight initialization specified
            Act::TanhW(weit) | Act::SigmoidW(weit) | Act::ReluW(weit) => weit,
            Act::Softmax | Act::LeakyRelu => todo!(),
        };

        LayerTerms::Dense(LayerOrder::FromSecondToLast, size, act, act_type, w_distr)
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
    type Output = (Vec<usize>, Vec<ActFp>, Vec<ActFp>, Act, Vec<Weit>);

    fn reduce(&self) -> Self::Output {
        let acc = (vec![0], vec![], vec![], Act::Sigmoid, vec![]); // acc is type Output
        self.0.iter().enumerate().fold(acc, |mut acc, (i,l)| {
            match l.reduce() {
                LayerTerms::Input(ref order, size) if order.valid(i) => {
                    acc.0.push(size); // input size
                    // swap_remove dummy value 0 with last element -> input size
                    acc.0.swap_remove(0);
                    acc
                },
                LayerTerms::Dense(ref order, size, act, act_type, weit) if order.valid(i) => {
                    let (act_fp, deriv_fp) = act.pair(); // activation and activation derivative functions
                    acc.0.push(size);
                    acc.1.push(act_fp);
                    acc.2.push(deriv_fp);
                    acc.3 = act_type;
                    acc.4.push(weit);
                    acc
                },
                _ => panic!("Layer order is incorrect, perhaps input layer is not first layer added?"),
            }
        })
    }
}
