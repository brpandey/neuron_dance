use crate::activation::functions::FunctionAct;

#[derive(Clone, Debug)]
pub struct Softmax;

impl FunctionAct for Softmax {

        def softmax(x):
    """Compute the softmax of vector x."""
        exps = np.exp(x)
        return exps / np.sum(exps)

    fn compute(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(x: f64) -> f64 {
        let s = Self::compute(x);
        s * (1.0 - s)
    }
}
