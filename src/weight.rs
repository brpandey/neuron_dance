use either::*;
use statrs::distribution::{Normal, Uniform};
use ndarray::Array2;
use ndarray_rand::RandomExt;

// Weight Initialization Types
#[derive(Copy, Clone, Debug)]
pub enum Weit {
    Default,
    He,
    GlorotN,
    GlorotU,
    NormalizedGlorot,
    LeCunn
}

impl Weit {
    pub fn random_distr(&self, fan_out: usize, fan_in: usize) -> Array2<f64> {
        let fan_avg = (fan_in+fan_out) as f64 / 2.0;

        let either = match self {
            // XavierN/GlorotN - Tanh, Sigmoid, Softmax
            Weit::GlorotN => {
                let glorot_term = 1.0/(fan_avg).sqrt();
                Left(Normal::new(0., glorot_term).unwrap())
            },
            Weit::GlorotU => {
                let glorot_term = 3.0/(fan_avg).sqrt();
                Right(Uniform::new(-glorot_term, glorot_term).unwrap())
            },
            Weit::NormalizedGlorot => {
                let glorot_term = (6.0_f64).sqrt()/((fan_in+fan_out) as f64).sqrt(); // ... uniform distribution
                Right(Uniform::new(-glorot_term, glorot_term).unwrap())
            },
            // He - ReLu, LeakyRelu, ELU, GELU, Swish, Mesh
            Weit::He => {
                let he_term = (2.0 / fan_in as f64).sqrt();
                Left(Normal::new(0., he_term).unwrap())
            },
            // LeCunn - SELU
            Weit::LeCunn => {
                let lecunn_term = 1.0/(fan_in as f64).sqrt();
                Left(Normal::new(0., lecunn_term).unwrap())
            },
            Weit::Default => unreachable!(),
        };

        match either {
            Left(l) => Array2::random((fan_out, fan_in), l),
            Right(r) => Array2::random((fan_out, fan_in), r),
        }
    }
}
