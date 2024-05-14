//use nanoserde::{DeJson, SerJson}; // tiny footprint and fast!
use nanoserde::{DeBin, SerBin}; // tiny footprint and fast!
use ndarray::{Array, Array2};
use std::{fs::File, fmt::Debug, path::Path};
use std::io::{Read, Write, Result};

use crate::{
    activation::{Act}, hypers::Hypers,
    optimizer::{Optimizer, Optim}, types::Batch,
};

pub const ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");

#[derive(Clone, Debug, Default, DeBin, SerBin)]
pub struct HypersArchive { // subset of Hypers
    pub learning_rate: f64,
    pub l2_rate: f64,
    pub class_size: usize,
    pub activations: Vec<String>, // Vec<String>, vec of activation strings
    pub loss_type: String, // loss as string type
    pub batch_type: String, // batch as string type
    pub optimizer_type: String, // optimizer as String type
}

#[derive(Clone, Debug, Default, DeBin, SerBin)]
pub struct VecArray2Archive<T> { // archive version of Array2 w/o pulling in ndarray serde features additional code
    pub shapes: Option<Vec<(usize, usize)>>,
    pub values: Option<Vec<Vec<T>>>
}


// Marker trait
pub trait Archive: Clone + Debug + Default + DeBin + SerBin {}

impl Archive for HypersArchive {}
impl<F64: Clone + Debug + Default + DeBin + SerBin> Archive for VecArray2Archive<F64> {}

pub trait Save {
    type Target: Archive;

    // define intermediate mapping to an archived version of type
    fn to_archive(&self) -> Self::Target;
    fn from_archive(archive: Self::Target) -> Self;

    // file save and restore methods which internally use
    // intermediate to and from archive mapping
    fn save<P: AsRef<Path> + std::fmt::Display>(&mut self, token: P) -> Result<()>
    where
        <Self as Save>::Target: SerBin
    {
        let path = format!("{}/saved_models/{}", ROOT_DIR, token);
        let archive = self.to_archive();
        let mut f = File::create(path)?;

        let bytes = SerBin::serialize_bin(&archive);
        f.write_all(&bytes[..])?;
        Ok(())
    }

    fn restore<P>(token: P) -> Self
    where
        P: AsRef<Path> + std::fmt::Display,
        Self: Sized,
        Self: Default,
        <Self as Save>::Target: DeBin
    {
        let path = format!("{}/saved_models/{}", ROOT_DIR, token);

        if let Ok(mut file) = File::open(path) {
            let mut buf = vec![];
            if file.read_to_end(&mut buf).is_ok() {
                if let Ok(archive) = DeBin::deserialize_bin(&buf) {
                    return Save::from_archive(archive)
                }
            }
        }

        Default::default()
    }
}


impl Save for Hypers {
    type Target = HypersArchive;

    fn to_archive(&self) -> Self::Target {
        dbg!(&self.batch_type);
        dbg!(&self.batch_type.to_string());

        HypersArchive {
            learning_rate: self.learning_rate,
            l2_rate: self.l2_rate,
            class_size: self.class_size,
            activations: self.activations.iter().map(|a| a.to_string()).collect(),
            loss_type: self.loss_type.to_string(),
            batch_type: self.batch_type.to_string(),
            optimizer_type: self.optimizer_type.to_string(),
        }
    }

    fn from_archive(archive: Self::Target) -> Self {
        let opt: Box<dyn Optimizer> =
            archive.optimizer_type.parse::<Optim>().unwrap().into();

        let acts = archive.activations
            .iter()
            .map(|act| act.parse().unwrap())
            .collect::<Vec<Act>>();

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

impl Save for Vec<Array2<f64>> { // custom trait for std lib type
    type Target = VecArray2Archive<f64>;

    fn to_archive(&self) -> Self::Target {
        let (mut values, mut shapes) = (vec![], vec![]);

        for v in self.iter() {
            values.push((*v).clone().into_raw_vec());
            shapes.push(v.dim());
        }

        VecArray2Archive{ shapes: Some(shapes), values: Some(values) }
    }

    fn from_archive(archive: Self::Target) -> Self {
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
