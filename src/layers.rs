use crate::activation::{Activation, MathFp};

// Make fields public for layers literal construction
pub struct Input1(pub usize);
pub struct Input2(pub (usize, usize));
pub struct Dense(pub usize, pub String);

pub trait Layer {
    type Output;

    fn reduce(&self) -> Self::Output;
}

pub enum LayerReduce {
    Input(usize),
    Dense(usize, Box<dyn Activation>),
}

impl Layer for Input1 {
    type Output = LayerReduce;

    fn reduce(&self) -> Self::Output {
        LayerReduce::Input(self.0)
    }
}

impl Layer for Input2 {
    type Output = LayerReduce;

    fn reduce(&self) -> Self::Output {
        LayerReduce::Input(self.0.0 * self.0.1)
    }
}

impl Layer for Dense {
    type Output = LayerReduce;

    fn reduce(&self) -> Self::Output {
        let size = self.0;
        let act: Box<dyn Activation> = self.1.as_str().parse().unwrap();
        LayerReduce::Dense(size, act)
    }
}

pub struct LayersGroup(Vec<Box<dyn Layer<Output = LayerReduce>>>); // holds a vec of layer trait objects

impl LayersGroup {
    pub fn new() -> Self {
        LayersGroup(vec![])
    }

    pub fn add<L: Layer<Output = LayerReduce> + 'static>(&mut self, l: L) {
        self.0.push(Box::new(l));
    }

    pub fn len(&self) -> usize { self.0.len() }
}

impl Layer for LayersGroup {
    //tuple of vecs: (sizes, MathFp, MathFp)
    type Output = (Vec<usize>, Vec<MathFp>, Vec<MathFp>);

    fn reduce(&self) -> Self::Output {
        let acc = (vec![0], vec![], vec![]); // acc is type Output
        self.0.iter().fold(acc, |mut acc, l| {
            match l.reduce() {
                LayerReduce::Input(size) => {
                    acc.0.push(size); // input size
                    // swap_remove dummy value 0 with last element -> input size
                    acc.0.swap_remove(0);
                    acc
                }
                LayerReduce::Dense(size, act) => {
                    let (act_fp, deriv_fp) = act.pair(); // activation and activation derivative functions
                    acc.0.push(size);
                    acc.1.push(act_fp);
                    acc.2.push(deriv_fp);
                    acc
                },
            }
        })
    }
}
