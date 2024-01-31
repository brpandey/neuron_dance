use std::iter::Iterator;
use ndarray::{Array, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
// use ndarray_rand::rand_distr::StandardNormal;
// use ndarray::arr2;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::Uniform;

use crate::activation::Activation;
use crate::algebra::*;
use crate::computation::{CacheComputation, ChainRuleComputation};
use crate::dataset::TrainTestSplitRef;

static SGD_EPOCHS: usize = 10000;
static MINIBATCH_EPOCHS: usize = 30;

pub struct Network {
    #[allow(dead_code)]
    sizes: Vec<usize>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    activations: Vec<Box<dyn Activation>>,
    learning_rate: f64,
    total_layers: usize,
}

impl Network {
    pub fn new(sizes: Vec<usize>, activations: Vec<Box<dyn Activation>>, learning_rate: f64) -> Self {
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
            weights.push(w);
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

    pub fn train_sgd(&mut self, test_train: TrainTestSplitRef) {
        let mut rng;
        let mut random_index;
        let (mut x_single, mut y_single);
        let mut cc = CacheComputation::new(&self.activations);

        let (train, test) = (&test_train.0, &test_train.1);

        for _ in 0..SGD_EPOCHS { // train and update network based on single observation sample
            rng = rand::thread_rng();
            random_index = rng.gen_range(0..train.size);
            x_single = train.x.select(Axis(0), &[random_index]); // arr2(&[[0.93333333, 0.93333333, 0.81960784]])
            y_single = train.y.select(Axis(0), &[random_index]); // arr2(&[[1.0]])

            self.train_iteration(x_single.t(), &y_single, self.learning_rate, &mut cc);
        }

        let result = self.evaluate(test.x, test.y, test.size);
        println!("Accuracy {:?} {}/{} {} (SGD)", result.0, result.1, result.2, SGD_EPOCHS);
    }

    pub fn train_minibatch(&mut self, test_train: TrainTestSplitRef, batch_size: usize) {
        let (mut x_minibatch, mut y_minibatch);
        let mut cc = CacheComputation::new(&self.activations);

        let (train, test) = (&test_train.0, &test_train.1);
        let mut row_indices = (0..train.size).collect::<Vec<usize>>();

        for e in 0..MINIBATCH_EPOCHS {
            row_indices.shuffle(&mut rand::thread_rng());

            for i in row_indices.chunks(batch_size) { //train and update network after each batch size of observation samples
                x_minibatch = train.x.select(Axis(0), &i);
                y_minibatch = train.y.select(Axis(0), &i);

                // transpose to ensure proper matrix multi fit
                self.train_iteration(x_minibatch.t(), &y_minibatch, self.learning_rate/batch_size as f64, &mut cc);
            }

            let result = self.evaluate(test.x, test.y, test.size);
            println!("Epoch {}: accuracy {:?} {}/{} {} (MiniBatch)", e, result.0, result.1, result.2, MINIBATCH_EPOCHS);
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
        // Z = W*A0 + B, A1 = RELU(Z) or A2 = Sigmoid(Z)
        for ((w, b), act) in self.weights.iter().zip(self.biases.iter()).zip(self.activations.iter()) {
            z = z_linear(w, &acc, b); // z = w.dot(&acc) + b;
            a = a_nonlinear(&mut z, act); // σ(z)

            acc = a;
            opt.as_mut().map(|cc| cc.cache(z, &acc));
        }

        acc // return last computed activation values
    }

    pub fn backward_pass<'b, 'c>(&self, y: &Array2<f64>, cc: &'b mut CacheComputation) -> ChainRuleComputation<'c>
    where 'b: 'c // cc is around longer than crc
    {
        // Compute the chain rule values for each layer
        // Store the partial cost derivative for biases and weights from each layer,
        // starting with last layer first

        let mut crc = ChainRuleComputation::new(cc);
        let acc0: Array2<f64> = crc.init(y);

        // zip number of iterations with corresponding weight (start from back to front layer)
        let zipped = (0..self.total_layers-2).zip(self.weights.iter().rev());
        zipped.fold(acc0, |acc: Array2<f64>, (_, w)| {
            crc.fold_layer(acc, w)
        });

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

    pub fn evaluate(&self, x_test: &Array2<f64>, y_test: &Array2<f64>, n_test: usize) -> (f64, usize, usize) {
        // run forward pass with no caching of intermediate values on each observation data
        let mut output: Array2<f64>;
        let mut empty: Option<&mut CacheComputation> = None;
        let mut matches: usize = 0;

        // processes an x_test row of input values at a time
        for (x_sample, y) in x_test.axis_chunks_iter(Axis(0), 1).zip(y_test.iter()) {
            output = self.predict(x_sample.t(), &mut empty);

//            println!("predict x_sample {} out {} am {} y {:?}", x_sample.view(), &output.view(), arg_max(&output), y);
            if arg_max(&output) == *y as usize {
                matches += 1;
            }
        }

        let accuracy = matches as f64 / n_test as f64;

        (accuracy, matches, n_test)
    }
}
