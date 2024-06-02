use crate::archive::Archive;
use crate::dataset::sanitize_token;
use crate::types::SimpleError;
/// Save trait
/// Defines a simple interface for binary file serialization and deserialization
/// using nanoserde instead of other libraries for its tiny dependency
/// footprint and speed

/// Defines an associated type Proxy which serves as a mapping between
/// type's structure and type's preference for its archived version
/// Could be same as type or different as long as there are conversion routines
use nanoserde::{DeBin, SerBin};
use std::{
    fs::File,
    io::{Read, Write},
};

pub const ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");

pub trait Save {
    type Proxy: Archive; // intermediate structure

    fn to_archive(&self) -> Self::Proxy
    where
        <Self as Save>::Proxy: for<'a> From<&'a Self>,
    {
        self.into()
    }

    fn from_archive(archive: Self::Proxy) -> Result<Self, SimpleError>
    where
        Self: Sized,
        Self: From<<Self as Save>::Proxy>,
    {
        Ok(archive.into())
    }

    // file save and restore methods which use
    // intermediate or proxy archive mapping
    fn save<S>(&self, token: S) -> Result<(), SimpleError>
    where
        S: AsRef<str>,
        <Self as Save>::Proxy: SerBin,
        Self: Sized,
        <Self as Save>::Proxy: for<'a> From<&'a Self>,
    {
        let tok = sanitize_token(token.as_ref())?;
        let path = format!("{}/saved_models/{}.txt", ROOT_DIR, tok);
        let archive = self.to_archive();
        let mut f = File::create(path)?;

        let bytes = SerBin::serialize_bin(&archive);
        f.write_all(&bytes[..])?;
        Ok(())
    }

    fn restore<S>(token: S) -> Result<Self, SimpleError>
    where
        S: AsRef<str>,
        <Self as Save>::Proxy: DeBin,
        Self: Sized,
        Self: From<<Self as Save>::Proxy>,
    {
        let tok = sanitize_token(token.as_ref())?;
        let path = format!("{}/saved_models/{}.txt", ROOT_DIR, tok);

        let mut file = File::open(path)?;
        let mut buf = vec![];
        file.read_to_end(&mut buf)?;

        let archive = DeBin::deserialize_bin(&buf)?;
        Save::from_archive(archive)
    }
}
