use std::iter::Iterator;
use std::collections::VecDeque;

use ndarray::{Array, Array2, Axis, ArrayView2};
use rand::distributions::Uniform;
// use ndarray_rand::rand_distr::StandardNormal;
use rand::Rng;
use ndarray_rand::{RandomExt};
use rand::seq::SliceRandom;

static EPOCHS: usize = 1000;

pub struct Network {
    #[allow(dead_code)]
    sizes: Vec<usize>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    activations: Vec<Functions>,
    learning_rate: f64,
    total_layers: usize,
}

impl Network {
    // e.g. [2,3,1] or [3,3,1]
    // Vec["relu", "sigmoid"]
    pub fn new(sizes: Vec<usize>, activations: Vec<Functions>, learning_rate: f64) -> Self {
        let (mut weights, mut biases) : (Vec<Array2<f64>>, Vec<Array2<f64>>) = (vec![], vec![]);
        let (mut x, mut y);
        let (mut b, mut w) : (Array2<f64>, Array2<f64>);
        let size = sizes.len();

        for i in 1..size {
            x = sizes[i-1];
            y = sizes[i];

            b = Array::random((y, 1), Uniform::new(0., 1.)); // for sizes [2,3,1] => 3x1 b1, b2, b3, and 1x1 b4
            w = Array::random((y, x), Uniform::new(0., 1.)); // for sizes [2,3,1] => 3*2, w1, w2, ... w5, w6..,
                                                             // 1*3, w7, w8, w9

            println!("random weight {:8.4}", &w);
            weights.push(w);

            println!("random biases {:8.4}", &b);
            biases.push(b);
        }

        Network {
            sizes,
            weights,
            biases,
            activations,
            learning_rate,
            total_layers: size,
        }
    }

    pub fn train_sgd(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, train_size: usize) {
        let mut rng;
        let mut random_index;
        let (mut x_single, mut y_single);
        for _ in 0..EPOCHS { // train and update network based on single observation sample
            rng = rand::thread_rng();
            random_index = rng.gen_range(0..train_size);
            x_single = x_train.select(Axis(0), &[random_index]);
            y_single = y_train.select(Axis(0), &[random_index]);

            self.train_iteration(x_single.t(), &y_single, self.learning_rate);
        }
    }

    pub fn train_minibatch(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, train_size: usize, batch_size: usize) {
        let mut row_indices = (0..train_size).collect::<Vec<usize>>();
        let (mut x_minibatch, mut y_minibatch);
        for _ in 0..EPOCHS {
            row_indices.shuffle(&mut rand::thread_rng());

            for i in row_indices.chunks(batch_size) { //train and update network after each batch size of observation samples
                x_minibatch = x_train.select(Axis(0), &i);
                y_minibatch = y_train.select(Axis(0), &i);

                // transpose to ensure proper matrix multi fit
                self.train_iteration(x_minibatch.t(), &y_minibatch, self.learning_rate/batch_size as f64);
            }
        }
        //            println!("Epoch {}: {} / {}", j, self.evaluate(test_data), n_test);

    }

    pub fn train_iteration(&mut self, x_iteration: ArrayView2<f64>, y_iteration: &Array2<f64>, learning_rate: f64) {
        let (mut z_values, mut a_values) = self.forward_pass(x_iteration); // transpose x single
        let deltas = self.backward_pass(&mut z_values, &mut a_values, y_iteration);
        self.update_iteration(deltas, learning_rate);
    }

    // forward pass is a wrapper around predict as it tracks the intermediate linear and non-linear values
    pub fn forward_pass(&self, x: ArrayView2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let (mut z_values, mut a_values) = (Vec::new(), vec![x.to_owned()]);

        // capture intermediate values
        let store = |z: Array2<f64>, a: &Array2<f64>| {
            z_values.push(z);
            a_values.push(a.to_owned());
        };

        let mut option_func = Some(Box::new(store));
        self.predict(x, &mut option_func);

        (z_values, a_values)
    }

    pub fn predict<F>(&self, x: ArrayView2<f64>, f: &mut Option<Box<F>>) -> Array2<f64>
    where
        F: FnMut(Array2<f64>, &Array2<f64>) + ?Sized
    {
        let mut z: Array2<f64>;
        let mut a: Array2<f64>;
        let mut acc = x.to_owned();

        // Compute and store the linear Z values and nonlinear A (activation) values
        // Z = W*X + B, A = RELU(Z), A = Sigmoid(Z) or Z = W*A + B values
        for ((w, b), act) in self.weights.iter().zip(self.biases.iter()).zip(self.activations.iter()) {
            z = w.dot(&acc) + b;
            a = Network::apply_nonlinear(&mut z, act);

            acc = a;
            //            f(z, &acc);
            f.as_mut().map(|fun| fun(z, &acc));
        }

        acc // return last computed activation values
    }

    pub fn backward_pass(&self, z_values: &mut Vec<Array2<f64>>, a_values: &mut Vec<Array2<f64>>, y: &Array2<f64>) -> (VecDeque<Array2<f64>>, VecDeque<Array2<f64>>) {
        // Store the partial cost derivative for biases and weights from each layer,
        // starting with last layer first

        let mut deltas: (VecDeque<Array2<f64>>, VecDeque<Array2<f64>>) = (VecDeque::new(), VecDeque::new()); // (bias_deltas, weight_deltas)

        // Reverse iterators
        let (mut funcs_riter, mut weights_riter) = (self.activations.iter().rev(),
                                                    self.weights.iter().rev());

        let a_last = a_values.pop().unwrap();
        let dc_da = Network::cost_derivative(&a_last, &y);
        let da_dz = Network::apply_nonlinear_derivative(z_values, &mut funcs_riter).unwrap();
        let mut acc: Array2<f64> =  dc_da * da_dz; // dC_dZ2 

        Network::calculate_deltas(&acc, a_values, (&mut deltas.0, &mut deltas.1));

        let mut w;
        let mut dzj_dai; // j is layer after the layer i, e.g. j is 2 and i is 1
        let mut dai_dzi;

        for _ in 0..self.total_layers-2 {
            w = weights_riter.next().unwrap();

            dzj_dai = w.t();
            dai_dzi = Network::apply_nonlinear_derivative(z_values, &mut funcs_riter).unwrap();

            // (dZ2_dA1 dot dC_dZ2) * dA1_dZ1 => dC_dZ1 or
            // (dzj_dai dot dC_dZj) * dai_dzi => dC_dZi where j is the layer after i
            acc = dzj_dai.dot(&acc) * dai_dzi;

            Network::calculate_deltas(&acc, a_values, (&mut deltas.0, &mut deltas.1));
        }

        return deltas
    }

    pub fn calculate_deltas(dc_dz: &Array2<f64>, a_values: &mut Vec<Array2 <f64>>,
                            deltas: (&mut VecDeque<Array2<f64>>, &mut VecDeque<Array2<f64>>)) {
        let dz_db = 1.0;
        let dc_db = dc_dz * dz_db;
        deltas.0.push_front(dc_db);

        let dz_dw = a_values.pop().unwrap();
        let dc_dw = dc_dz.dot(&dz_dw.t());
        deltas.1.push_front(dc_dw);
    }

    pub fn update_iteration(&mut self, deltas: (VecDeque<Array2<f64>>, VecDeque<Array2<f64>>), learning_rate: f64) {
        let bias_deltas = deltas.0;
        let weight_deltas = deltas.1;

        for (b, db) in self.biases.iter_mut().zip(bias_deltas.iter()) {
            //            *b -= self.learning_rate * db
            *b -= &db.mapv(|x| x * learning_rate)
        }

        for (w, dw) in self.weights.iter_mut().zip(weight_deltas.iter()) {
//            *w -= self.learning_rate * dw
            *w -= &dw.mapv(|x| x * learning_rate)
        }

    }

    pub fn evaluate(&self, x_test: &Array2<f64>, y_test: &Array2<f64>, n_test: usize) {
        // run forward pass on each observation data
        let _samples: Vec<usize> = (0..n_test).collect();
        let mut _output: Array2<f64>;
        let mut _x_single: ArrayView2<f64>;

        /*
        for s in 0..n_test {
        x_single = x_test.select(Axis(0), &[s]);
            output = self.predict(&x_single, nostore);
        if Self::arg_max(&output) == *y_test.get((s,0)).unwrap() as usize { // classification
                matches += 1;
    }
        }
         */

        /*
        output = self.predict(x_test);
        output2 = Self::arg_max(&output);
        for (x, y) in output2.iter().zip(y_test.iter()) {
        if x == y {
                matches += 1;
    }
        }
         */

        /*
        let b = arr2(&[[11, 12, 13],
        [14, 15, 16]]);

        let mut iter = b.axis_chunks_iter(Axis(0), 1);
        let x = iter.next().unwrap();
        assert_eq!(x, arr2(&[[11, 12, 13]]));
        */
//        let no_function = None;
        let mut empty: Option<Box<dyn FnMut(Array2<f64>, &Array2<f64>)>> = None;
        let mut predictions = vec![];

        for x_sample in x_test.axis_chunks_iter(Axis(0), 1) {
            let output = self.predict(x_sample, &mut empty);
            predictions.push(Network::arg_max(&output));
        }

        let matches: usize = predictions.iter().zip(y_test.iter())
            .map(|(x, y)| (*x == *y as usize) as usize)
            .sum();

        println!("EPOCH X matches {matches} / n_test {n_test}, ACCURACY {:?}", matches/n_test);
    }

    pub fn arg_max(output: &Array2<f64>) -> usize {
        let mut max_acc_index = 0;

        // if we have a single neuron output, return either 0 or 1
        if output.shape() == &[1,1] {
            return output[[0, 0]].round() as usize;
        }

        // Find the index of the output neuron with the highest activation
        for (i, &v) in output.iter().enumerate() {
            if v > output[[0, max_acc_index]] { // compare value from first row (0) of 2d array by index
                max_acc_index = i;
            }
        }

        max_acc_index
    }

    pub fn apply_nonlinear(z: &mut Array2<f64>, func_type: &Functions) -> Array2<f64> {
        z.mapv(|v| Activations::apply(func_type, v))
    }

    pub fn apply_nonlinear_derivative<'a, I>(z_values: &mut Vec<Array2<f64>>, rev_func_names: &mut I) -> Option<Array2<f64>>
    where
        I: Iterator<Item = &'a Functions>
    {

        if let (Some(z_last), Some(func_name)) = (z_values.pop(), rev_func_names.next()) {
            let da_dz = z_last.mapv(|v| Activations::apply_derivative(func_name, v));
            return Some(da_dz)
        }

        None
    }

    /// Assuming cost is (a - y)^2
    pub fn cost_derivative(output_a: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        2.0*(output_a - y)
    }
}


#[derive(Debug, Copy, Clone)]
pub enum Functions {
    Sigmoid,
    Relu,
}

pub struct Activations;

impl Activations {

    pub fn apply(name: &Functions, x: f64) -> f64 {
        match name {
            Functions::Sigmoid => Activations::sigmoid(x),
            Functions::Relu => Activations::relu(x),
        }
    }

    pub fn apply_derivative(name: &Functions, x: f64) -> f64 {
        match name {
            Functions::Sigmoid => Activations::sigmoid_derivative(x),
            Functions::Relu => Activations::relu_derivative(x),
        }
    }

    // Activation functions and their derivatives
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn sigmoid_derivative(x: f64) -> f64 {
        Activations::sigmoid(x) * (1.0 - Activations::sigmoid(x))
    }

    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    pub fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}
