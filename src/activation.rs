#[derive(Debug, Copy, Clone)]
pub enum Function {
    Sigmoid,
    Relu,
}

pub struct Activations;

impl Activations {
    pub fn apply(name: &Function, x: f64) -> f64 {
        match name {
            Function::Sigmoid => Activations::sigmoid(x),
            Function::Relu => Activations::relu(x),
        }
    }

    pub fn apply_derivative(name: &Function, x: f64) -> f64 {
        match name {
            Function::Sigmoid => Activations::sigmoid_derivative(x),
            Function::Relu => Activations::relu_derivative(x),
        }
    }

    // Activation functions and their derivatives
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn sigmoid_derivative(x: f64) -> f64 {
        let s = Activations::sigmoid(x);
        s * (1.0 - s)
    }

    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    pub fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

