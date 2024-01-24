use std::iter::Iterator;

use ndarray::{Array, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
// use ndarray_rand::rand_distr::StandardNormal;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::Uniform;

use crate::activation::Function;
use crate::computation::{CacheComputation, ChainRuleComputation};
use crate::algebra::{arg_max, apply_linear, apply_nonlinear};

static SGD_EPOCHS: usize = 10000;
static MINIBATCH_EPOCHS: usize = 30;

pub struct Network {
    sizes: Vec<usize>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    activations: Vec<Function>,
    learning_rate: f64,
    #[allow(dead_code)]
    total_layers: usize,
}

impl Network {
    // e.g. [2,3,1] or [3,3,1]
    // Vec["relu", "sigmoid"]
    pub fn new(sizes: Vec<usize>, activations: Vec<Function>, learning_rate: f64) -> Self {
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

    pub fn train_sgd(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, train_size: usize,
                     x_test: &Array2<f64>, y_test: &Array2<f64>, test_size: usize) {
        let mut rng;
        let mut random_index;
        let (mut x_single, mut y_single);
        let mut cc = CacheComputation::new(self.activations.clone());

        for _ in 0..SGD_EPOCHS { // train and update network based on single observation sample
            rng = rand::thread_rng();
            random_index = rng.gen_range(0..train_size);
            x_single = x_train.select(Axis(0), &[random_index]);
            y_single = y_train.select(Axis(0), &[random_index]);

            self.train_iteration(x_single.t(), &y_single, self.learning_rate, &mut cc);
        }

        let result = self.evaluate(x_test, y_test, test_size);
        println!("Accuracy {:?} {}", result, test_size);
    }

    pub fn train_minibatch(&mut self, x_train: &Array2<f64>, y_train: &Array2<f64>, train_size: usize, batch_size: usize,
                           x_test: &Array2<f64>, y_test: &Array2<f64>, test_size: usize) {
        let mut row_indices = (0..train_size).collect::<Vec<usize>>();
        let (mut x_minibatch, mut y_minibatch);
        let mut cc = CacheComputation::new(self.activations.clone());

        for e in 0..MINIBATCH_EPOCHS {
            row_indices.shuffle(&mut rand::thread_rng());

            for i in row_indices.chunks(batch_size) { //train and update network after each batch size of observation samples
                x_minibatch = x_train.select(Axis(0), &i);
                y_minibatch = y_train.select(Axis(0), &i);

                // transpose to ensure proper matrix multi fit
                self.train_iteration(x_minibatch.t(), &y_minibatch, self.learning_rate/batch_size as f64, &mut cc);
            }

            let result = self.evaluate(x_test, y_test, test_size);
            println!("Epoch {}: accuracy {:?}  test_size {}", e, result, test_size);
        }
    }

    pub fn train_iteration(&mut self, x_iteration: ArrayView2<f64>, y_iteration: &Array2<f64>, learning_rate: f64, cc: &mut CacheComputation) {
        self.forward_pass(x_iteration, cc);
        let chain_rule_compute = self.backward_pass(y_iteration, cc);
        self.update_iteration(learning_rate, chain_rule_compute);
    }

    // forward pass is a wrapper around predict as it tracks the intermediate linear and non-linear values
    pub fn forward_pass(&self, x: ArrayView2<f64>, cc: &mut CacheComputation) {
        cc.init(x.to_owned());
        let mut opt_comp = Some(cc);
        self.predict(x, &mut opt_comp);
    }

    pub fn predict(&self, x: ArrayView2<f64>, opt: &mut Option<&mut CacheComputation>) -> Array2<f64> {
        let mut z: Array2<f64>;
        let mut a: Array2<f64>;
        let mut acc = x.to_owned();

        // Compute and store the linear Z values and nonlinear A (activation) values
        // Z = W*X + B, A = RELU(Z), A = Sigmoid(Z) or Z = W*A + B values
        for ((w, b), act) in self.weights.iter().zip(self.biases.iter()).zip(self.activations.iter()) {
            z = apply_linear(w, &acc, b); // z = w.dot(&acc) + b;
            a = apply_nonlinear(&mut z, act); // σ(z)

            acc = a;
            opt.as_mut().map(|cc| cc.store_intermediate(z, &acc));
        }

        acc // return last computed activation values
    }

    pub fn backward_pass<'a, 'b, 'c>(&'a self, y: &Array2<f64>, cc: &'b mut CacheComputation) -> ChainRuleComputation<'c>
    where 'b: 'c // cc is around longer than crc
    {
        // Compute the chain rule values for each layer
        // Store the partial cost derivative for biases and weights from each layer,
        // starting with last layer first

        let mut crc = ChainRuleComputation::new(cc);
        let acc0: Array2<f64> = crc.init(y);
        self.weights.iter().rev().fold(acc0, |acc: Array2<f64>, w| crc.fold_layer(acc, w));

        crc
    }

    pub fn update_iteration(&mut self, learning_rate: f64, crc: ChainRuleComputation) {
        // Apply delta contributions to current biases and weights by subtracting
        // since we are taking the negative gradient using the chain rule to find a local
        // minima in our neural network cost graph as opposed to maxima (positive gradient)

        // Intuitively, if the deltas are negative, invert their sign and add them
        // if the deltas are positive, invert their sign and subtract them
        let (b_deltas, w_deltas) = (crc.bias_deltas(), crc.weight_deltas());

        for (b, db) in self.biases.iter_mut().zip(b_deltas) {
            *b -= &db.mapv(|x| x * learning_rate)
        }

        for (w, dw) in self.weights.iter_mut().zip(w_deltas) {
            *w -= &dw.mapv(|x| x * learning_rate)
        }
    }

    pub fn evaluate(&self, x_test: &Array2<f64>, y_test: &Array2<f64>, n_test: usize) -> f64 {
        // run forward pass with no caching of intermediate values on each observation data
        let mut output: Array2<f64>;

        /*
        let b = arr2(&[[11, 12, 13],
        [14, 15, 16]]);

        let mut iter = b.axis_chunks_iter(Axis(0), 1);
        let x = iter.next().unwrap();
        assert_eq!(x, arr2(&[[11, 12, 13]]));
        */

        let mut empty: Option<&mut CacheComputation> = None;
        let mut predictions = vec![];

        for x_sample in x_test.axis_chunks_iter(Axis(0), 1) {
            output = self.predict(x_sample, &mut empty);
            predictions.push(arg_max(&output));
        }

        let matches: usize =
            predictions.iter().zip(y_test.iter())
            .map(|(x, y)| if *x == *y as usize { 1 } else { 0 })
            .sum();

        let accuracy = matches as f64 / n_test as f64;
        println!("matches {matches} / n_test {n_test}, ACCURACY {:?}", accuracy);
        accuracy
    }
}





