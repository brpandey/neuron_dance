use either::*;
use nanoserde::{DeBin, SerBin};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use statrs::distribution::{Normal, Uniform};
/// Weit or Weight Initialization Types
/// Provides layer specific initialization strategies to avoid vanishing or
/// exploding gradients from too large of weights or too small weights that
/// puts theh product of the weight and input scalar to close to zero or a large
/// saturating value where the derivative or rate of change becomes very large
/// (unstable) or very small (not enough contribution)

/// Initializing the weights randomly allows each
/// neuron to decide individually

/// Otherwise, using identical weight initialization values
/// ensures each layer's neurons will tend to make the same decision
/// repeatedly (same outputs)

/// Specifying a distribution with a mean value uniformly centers the random
/// generated scalars around the mean (e.g. 0.0) while keeping a smaller variance
/// e.g. 1/layer_size (especially for bigger layers) to keep the spread smaller
/// ensuring they group closer to each other

/// https://en.wikipedia.org/wiki/Variance

/// Normal distributions have a bell shape with values near the center more likely
/// than at the edges (e.g. shoe sizes). Whereas Uniform distributions are equally
/// spread out the data in more of a rectangular shape (e.g. rolling a dice)
use std::default::Default;

// Weight Initialization Types
#[derive(Copy, Clone, Debug, Default, SerBin, DeBin, PartialEq)]
pub enum Weit {
    #[default]
    Default,
    He,
    GlorotN,
    GlorotU,
    NormalizedGlorot,
    LeCunn,
}

impl Weit {
    pub fn random(&self, fan_out: usize, fan_in: usize) -> Array2<f64> {
        let fan_avg = (fan_in + fan_out) as f64 / 2.0;

        let either = match self {
            Weit::GlorotN => {
                let glorot_term = 1.0 / (fan_avg).sqrt();
                Left(Normal::new(0., glorot_term).unwrap())
            }
            // XavierN/GlorotU - Tanh, Sigmoid, Softmax
            Weit::GlorotU => {
                let glorot_term = 3.0 / (fan_avg).sqrt();
                Right(Uniform::new(-glorot_term, glorot_term).unwrap())
            }
            Weit::NormalizedGlorot => {
                let glorot_term = (6.0_f64).sqrt() / ((fan_in + fan_out) as f64).sqrt(); // ... uniform distribution
                Right(Uniform::new(-glorot_term, glorot_term).unwrap())
            }
            // He - ReLu, LeakyRelu, ELU, GELU, Swish, Mesh
            Weit::He => {
                let he_term = (2.0 / fan_in as f64).sqrt();
                Left(Normal::new(0., he_term).unwrap())
            }
            // LeCunn - SELU
            Weit::LeCunn => {
                let lecunn_term = 1.0 / (fan_in as f64).sqrt();
                Left(Normal::new(0., lecunn_term).unwrap())
            }
            Weit::Default => Left(Normal::new(0., 1.).unwrap()), // StandardNormal
        };

        match either {
            Left(l) => Array2::random((fan_out, fan_in), l),
            Right(r) => Array2::random((fan_out, fan_in), r),
        }
    }
}
