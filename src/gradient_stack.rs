use crate::activation::ActFp;
use ndarray::Array2;

// GradientStack is a collection of term stacks used primarily
// while computing the chain rule during back propagation

#[derive(Debug)]
pub struct GradientStack {
    linear: Vec<Array2<f64>>,    // linear values - z_values
    nonlinear: Vec<Array2<f64>>, // non-linear activation values - a_values
    funcs: Vec<ActFp>,           // activation derivative functions
    shapes: Vec<(usize, usize)>, // bias shapes
    index: (usize, usize),       // function, bias shape
}

impl GradientStack {
    pub fn new(backward: Vec<ActFp>, biases: &[Array2<f64>]) -> Self {
        // Compute bias shapes
        let shapes: Vec<(usize, usize)> = biases
            .iter()
            .map(|b| (b.shape()[0], b.shape()[1]))
            .collect();

        GradientStack {
            linear: vec![],
            nonlinear: vec![],
            funcs: backward, // activation derivative functions (backprop)
            shapes,
            index: (0, 0),
        }
    }

    // Reset stack values
    pub fn reset(&mut self, x: Array2<f64>) {
        (self.linear, self.nonlinear) = (Vec::new(), vec![x]);
        self.index = (self.funcs.len() - 1, self.shapes.len() - 1);
    }

    pub fn push(&mut self, kind: GT, data: Array2<f64>) {
        match kind {
            GT::Linear => self.linear.push(data),
            GT::Nonlinear => self.nonlinear.push(data),
            _ => (),
        }
    }

    pub fn pop(&mut self, kind: GT) -> Term {
        match kind {
            GT::Linear => Term::Linear(self.linear.pop()),
            GT::Nonlinear => Term::Nonlinear(self.nonlinear.pop()),
            GT::ActivationDerivative => {
                let f = self.funcs.get(self.index.0).unwrap();
                if self.index.0 != 0 {
                    self.index.0 -= 1;
                }
                Term::ActivationDerivative(f)
            }
            GT::BiasShape => {
                let s = self.shapes.get(self.index.1).unwrap();
                if self.index.1 != 0 {
                    self.index.1 -= 1;
                }
                Term::BiasShape(s.0, s.1)
            }
            _ => unreachable!(),
        }
    }
}

// Gradient Token - Input tokens describing types used for gradient calculation
pub enum GT {
    Linear,    // Z values
    Nonlinear, // A values
    ActivationDerivative,
    BiasShape,
    Features, // X values
}

// Single unified term contains a few variants depicting the items in the collective stack
pub enum Term<'a> {
    Linear(Option<Array2<f64>>),
    Nonlinear(Option<Array2<f64>>),
    ActivationDerivative(&'a ActFp),
    BiasShape(usize, usize),
}

// Helper methods to retrieve various data values
impl<'a> Term<'a> {
    pub fn array(self) -> Array2<f64> {
        match self {
            Term::Linear(Some(array2)) => array2,
            Term::Nonlinear(Some(array2)) => array2,
            _ => panic!("mismatch types, or no value found"),
        }
    }

    pub fn shape(self) -> (usize, usize) {
        if let Term::BiasShape(x, y) = self {
            (x, y)
        } else {
            (0, 0)
        }
    }

    pub fn fp(self) -> &'a ActFp {
        if let Term::ActivationDerivative(fp) = self {
            fp
        } else {
            panic!("mismatched term types")
        }
    }
}
