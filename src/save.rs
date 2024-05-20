use nanoserde::{DeBin, SerBin}; // tiny footprint and fast!
use ndarray::{Array, Array2};
use std::{fs::File, fmt::Debug, path::Path};
use std::io::{Read, Write};
use crate::types::SimpleError;

pub const ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");

// Marker trait
pub trait Archive: Clone + Debug + Default + DeBin + SerBin {}

pub trait Save {
    type Target: Archive;

    // define intermediate mapping to an archived version of type
    fn to_archive(&self) -> Self::Target;
    fn from_archive(archive: Self::Target) -> Self;

    // file save and restore methods which internally use
    // intermediate to and from archive mapping
    fn save<P: AsRef<Path> + std::fmt::Display>(&mut self, token: P) -> Result<(), SimpleError>
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

    fn restore<P>(token: P) -> Result<Self, SimpleError>
    where
        P: AsRef<Path> + std::fmt::Display,
        Self: Sized,
        <Self as Save>::Target: DeBin
    {
        let path = format!("{}/saved_models/{}", ROOT_DIR, token);

        let mut file = File::open(path)?;
        let mut buf = vec![];
        file.read_to_end(&mut buf)?;
        let archive = DeBin::deserialize_bin(&buf)?;
        Ok(Save::from_archive(archive))
    }
}

#[derive(Clone, Debug, Default, DeBin, SerBin)]
pub struct VecArray2Archive<T> { // archive version of Array2 w/o pulling in ndarray serde features additional code
    pub shapes: Option<Vec<(usize, usize)>>,
    pub values: Option<Vec<Vec<T>>>
}

impl<F64: Clone + Debug + Default + DeBin + SerBin> Archive for VecArray2Archive<F64> {}

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
