use either::*;
use std::{ops::RangeBounds, convert::Into, fmt, fmt::Debug};
use ndarray::{Array, Array2};

use crate::activation::{Activation, ActFp};

// Make fields public for layers literal construction

pub struct Input1(pub usize);
pub struct Input2(pub usize, pub usize);
pub struct Dense(pub usize, pub Act);

// re-export types into Layer, to consolidate interface
pub use crate::activation::Act;
pub use crate::cost::Loss;
pub use crate::types::{Batch, Eval, Metr, SimpleError};
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
        let mut act_type = self.1;
        let act: Box<dyn Activation> = self.1.into();

        let w_distr = match act_type {
            // act functions w default weight initializations
            Act::Softmax | Act::Tanh | Act::Sigmoid => Weit::GlorotU,
            Act::Relu => Weit::He,
            // act functions with weight initialization specified
            Act::Tanh_(weit) | Act::Sigmoid_(weit) | Act::Softmax_(weit) | Act::Relu_(weit) => weit,
            Act::LeakyRelu => todo!(),
        };

        act_type = act_type.normalize();
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
    // ((input_size,last_size), weights, biases, forward, backward, output_act)
    type Output = ((usize,usize), Vec<Array2<f64>>, Vec<Array2<f64>>, Vec<ActFp>, Vec<ActFp>, Vec<Act>);

    fn reduce(&self) -> Self::Output {
        use statrs::distribution::Normal;
        use ndarray_rand::RandomExt;

        let acc = ((0,0), vec![], vec![], vec![], vec![], vec![]); // acc is of type Output

        self.0.iter().enumerate().fold(acc, |mut acc, (i,l)| {
            match l.reduce() {
                LayerTerms::Input(ref order, size) if order.valid(i) => {
                    acc.0 = (size, size); // save input size as last size
                    acc
                },
                LayerTerms::Dense(ref order, size, act, act_type, weit) if order.valid(i) => {
                    let (act_fp, act_deriv_fp) = act.pair(); // activation and activation derivative functions

                    let x = acc.0.1; // (previous size) current layer inputs
                    let y = size; // current layer neurons (outputs)
                    acc.0.1 = size; // save last size, which at end will be output_size

                    // Note: z = wx + b, w is on left and x is transposed from csv row into vertical collumn
                    let b = Array::random((y, 1), Normal::new(0., 1.).unwrap()); // for sizes [2,3,1] => 3x1 b1, b2, b3, and 1x1 b4
                    let w = weit.random(y, x); // for sizes [2,3,1] => 3*2, w1, w2, ... w5, w6..,

                    acc.1.push(w); // push to weights
                    acc.2.push(b); // push to biases
                    acc.3.push(act_fp); // push to forward activations
                    acc.4.push(act_deriv_fp); // push to backward, or derivative activations
                    acc.5.push(act_type);
                    acc
                },
                _ => panic!("Layer order is incorrect, perhaps input layer is not first layer added?"),
            }
        })
    }
}
