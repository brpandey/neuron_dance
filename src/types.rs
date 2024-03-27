use std::fmt;

// Types not exclusive to any module

#[derive(Debug, Copy, Clone)]
pub enum Eval { Train, Test }

#[derive(Debug, Copy, Clone)]
pub enum Batch { SGD, Mini(usize) }

impl Batch {
    pub fn value(&self) -> usize {
        match self {
            Batch::SGD => 1,
            Batch::Mini(ref size) => *size,
        }
    }
}

impl fmt::Display for Batch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Batch::SGD => write!(f, "(SGD)"),
            Batch::Mini(_) => write!(f, "(MiniBatch)"),
        }
    }
}


#[derive(Debug)]
pub enum Classification {
    Binary,
    MultiClass,
}
