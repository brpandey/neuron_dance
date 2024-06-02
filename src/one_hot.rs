use ndarray::{ArrayBase, DataMut, DataOwned, Dimension, IntoDimension};
use num::{Float, Zero};
use std::collections::HashMap;
use std::{clone::Clone, marker::Copy};

pub fn one_hot<S, D, I, T>(shape: I, index: I) -> HashMap<usize, ArrayBase<S, D>>
where
    S: DataOwned<Elem = T> + DataMut,
    T: Clone + Zero + Float,
    D: Dimension + Copy,
    I: IntoDimension<Dim = D>,
{
    let mut one_hot = HashMap::new();
    let mut zeros;

    let sh: D = shape.into_dimension();
    let mut idx: D = index.into_dimension();
    let output_size = sh[0];

    for i in 0..output_size {
        zeros = ArrayBase::zeros(sh);
        idx[0] = i;
        *zeros.get_mut(idx).unwrap() = T::from(1.0).unwrap();
        one_hot.insert(i, zeros);
    }

    one_hot
}
