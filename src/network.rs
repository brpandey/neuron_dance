use std::iter::Iterator;
use ndarray::{Array, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use rand::{Rng, seq::SliceRandom};
use statrs::distribution::Normal;

use crate::{activation::MathFp, algebra::Algebra}; // import local traits
use crate::cache_computation::CacheComputation;
use crate::chain_rule::ChainRuleComputation;
use crate::dataset::TrainTestSubsetRef;
use crate::layers::{Layer, LayerStack, LayerTerms};

extern crate blas_src; // C & Fortran linear algebra library for optimized matrix compute

static SGD_EPOCHS: usize = 20000;
static MINIBATCH_EPOCHS: usize = 20;

pub struct Network {
    output_size: usize,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    forward: Vec<MathFp>, // forward propagation, activation functions
    backward: Vec<MathFp>, // backward propagation, activation derivatives
    learning_rate: f64,
    layers: Option<LayerStack>,
    total_layers: usize,
}

impl Network {
    pub fn new() -> Self { Self::empty() }

    fn empty() -> Self {
        Network {
            output_size: 0, weights: vec![], biases: vec![],
            forward: vec![], backward: vec![], learning_rate: 0.0,
            layers: Some(LayerStack::new()), total_layers: 0,
        }
    }

    pub fn add<L: Layer<Output = LayerTerms> + 'static>(&mut self, layer: L) {
        self.layers.as_mut().unwrap().add(layer);
    }

    //     "quadratic_cost", "adam", 0.2, "loss, accuracy";
    pub fn compile(&mut self, learning_rate: f64) { 
        let (sizes, forward, backward) = self.layers.as_mut().unwrap().reduce();
        let total_layers = self.layers.as_ref().unwrap().len();

        let (mut weights, mut biases) : (Vec<Array2<f64>>, Vec<Array2<f64>>) = (vec![], vec![]);
        let (mut x, mut y);
        let (mut b, mut w) : (Array2<f64>, Array2<f64>);

        for i in 1..total_layers {
            x = sizes[i-1];
            y = sizes[i];

            b = Array::random((y, 1), Normal::new(0., 1.).unwrap()); // for sizes [2,3,1] => 3x1 b1, b2, b3, and 1x1 b4
            w = Array::random((y, x), Normal::new(0., 1.).unwrap()); // for sizes [2,3,1] => 3*2, w1, w2, ... w5, w6..,

            weights.push(w);
            biases.push(b);
        }

        // replace empty network with new initialized network
        let n = Network {
            output_size: sizes[total_layers-1],
            weights,
            biases,
            forward,
            backward,
            learning_rate,
            layers: self.layers.take(),
            total_layers,
        };

        let _ = std::mem::replace(self, n);
    }

    pub fn train_sgd(&mut self, subsets: TrainTestSubsetRef) {
        let mut rng;
        let mut random_index;
        let (mut x_single, mut y_single);
        let mut cc = CacheComputation::new(&self.backward, &self.biases, self.output_size, 1);

        let (train, test) = (&subsets.0, &subsets.1);

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

    pub fn train_minibatch(&mut self, subsets: TrainTestSubsetRef, batch_size: usize) {
        let (mut x_minibatch, mut y_minibatch);
        let mut cc = CacheComputation::new(&self.backward, &self.biases, self.output_size, batch_size);

        let (train, test) = (&subsets.0, &subsets.1);
        let mut row_indices = (0..train.size).collect::<Vec<usize>>();

        for e in 0..MINIBATCH_EPOCHS {
            row_indices.shuffle(&mut rand::thread_rng());

            for c in row_indices.chunks(batch_size) { //train and update network after each batch size of observation samples
                x_minibatch = train.x.select(Axis(0), &c);
                y_minibatch = train.y.select(Axis(0), &c);

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
        self.update_iteration(chain_rule_compute, learning_rate);
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
        for ((w, b), a_func) in self.weights.iter().zip(self.biases.iter()).zip(self.forward.iter()) {
            z = acc.weighted_sum(w, b); // linear, z = w.dot(&acc) + b
            a = z.activate(a_func); // non-linear, σ(z)

            acc = a;
            opt.as_mut().map(|cc| cc.store(z, &acc));
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

    pub fn update_iteration(&mut self, crc: ChainRuleComputation, learning_rate: f64) {
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

            if output.arg_max() == *y as usize {
                matches += 1;
            }
        }

        let accuracy = matches as f64 / n_test as f64;

        (accuracy, matches, n_test)
    }
}
