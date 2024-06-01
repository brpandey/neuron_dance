use ndarray::{Array, Array2};
use nanoserde::{DeBin, SerBin}; // tiny footprint and fast!
use std::fmt::Debug;

use crate::{save::Save, network::Network};

// Marker trait, ensures archive structure can be serialized and deserialized
pub trait Archive: Clone + Debug + Default + DeBin + SerBin {}

// 1) VecArray2Archive
// Note: ndarray has a serde feature but this is simpler for now

#[derive(Clone, Debug, Default, DeBin, SerBin)]
pub struct VecArray2Archive<T> {
    pub shapes: Option<Vec<(usize, usize)>>,
    pub values: Option<Vec<Vec<T>>>
}

impl From<VecArray2Archive<f64>> for Vec<Array2<f64>> {
    fn from(archive: VecArray2Archive<f64>) -> Self {
        let (mut a, mut vec) = (archive, vec![]);
        let values_iter = a.values.take().unwrap().into_iter();
        let shapes_iter = a.shapes.take().unwrap().into_iter();

        for (v, s) in values_iter.zip(shapes_iter) {
            let array = Array::from_shape_vec(s, v).unwrap();
            vec.push(array);
        }

        vec
    }
}

impl From<&Vec<Array2<f64>>> for VecArray2Archive<f64> {
    fn from(vec: &Vec<Array2<f64>>) -> Self {
        let (mut values, mut shapes) = (vec![], vec![]);

        for v in vec.iter() {
            values.push((*v).clone().into_raw_vec());
            shapes.push(v.dim());
        }

        VecArray2Archive{ shapes: Some(shapes), values: Some(values) }
    }
}

impl<F64: Clone + Debug + Default + DeBin + SerBin> Archive for VecArray2Archive<F64> {}

impl Save for Vec<Array2<f64>> {
    type Proxy = VecArray2Archive<f64>;
}

// 2) Network Archive

#[derive(Clone, Debug, Default, DeBin, SerBin)]
pub struct NetworkArchive {
    pub weights: Option<VecArray2Archive<f64>>,
    pub biases: Option<VecArray2Archive<f64>>,
}

impl Archive for NetworkArchive {}

impl From<&Network> for NetworkArchive {
    fn from(network: &Network) -> Self {
        NetworkArchive {
            weights: Some(network.weights().into()),
            biases: Some(network.biases().into()),
        }
    }
}
