use std::{iter::Iterator, ops::{SubAssign, MulAssign}, fmt::Debug};
use ndarray::{Array, Array2, ArrayView2, Axis, ScalarOperand};
use ndarray_rand::{RandomExt, rand_distr::uniform::SampleUniform}; // rand_distr::StandardNormal;
use rand::{Rng, seq::SliceRandom, distributions::Uniform};
use num::Float;

use crate::{activation::{Activation, MathFp}, algebra::Algebra}; // import local traits
use crate::computation::{CacheComputation, ChainRuleComputation};
use crate::dataset::TrainTestSplitRef;

static SGD_EPOCHS: usize = 10000;
static MINIBATCH_EPOCHS: usize = 30;

pub struct Network<T> {
    #[allow(dead_code)]
    sizes: Vec<usize>,
    weights: Vec<Array2<T>>,
    biases: Vec<Array2<T>>,
    activations: Vec<Box<dyn Activation<T>>>,
    forward: Vec<MathFp<T>>, // forward activation functions
    learning_rate: T,
    total_layers: usize,
}

impl<T: Float + SampleUniform + ScalarOperand + SubAssign + MulAssign + Debug> Network<T> {
    pub fn new(sizes: Vec<usize>, layers: Vec<&str>, learning_rate: T) -> Self {
        let (mut weights, mut biases) : (Vec<Array2<T>>, Vec<Array2<T>>) = (vec![], vec![]);
        let (mut x, mut y);
        let (mut b, mut w) : (Array2<T>, Array2<T>);
        let size = sizes.len();

        // Construct activation function trait objects from supplied strs
        let activations = layers.into_iter()
            .map(|l| l.parse().unwrap())
            .collect::<Vec<Box<dyn Activation<T>>>>();

        // Create activation function collection given activation trait objects
        let forward: Vec<MathFp<T>> =
            activations.iter().map(|a| { let (c, _) = a.pair(); c}).collect();

        for i in 1..size {
            x = sizes[i-1];
            y = sizes[i];

            b = Array::random((y, 1), Uniform::new(num::zero::<T>(), num::one::<T>())); // for sizes [2,3,1] => 3x1 b1, b2, b3, and 1x1 b4
            w = Array::random((y, x), Uniform::new(num::zero::<T>(), num::one::<T>())); // for sizes [2,3,1] => 3*2, w1, w2, ... w5, w6..,
                                                             // 1*3, w7, w8, w9
            weights.push(w);
            biases.push(b);
        }

        Network {
            sizes,
            weights,
            biases,
            activations,
            forward,
            learning_rate,
            total_layers: size,
        }
    }

    pub fn train_sgd(&mut self, test_train: TrainTestSplitRef<T>) {
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

    pub fn train_minibatch(&mut self, test_train: TrainTestSplitRef<T>, batch_size: usize) {
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
                self.train_iteration(x_minibatch.t(), &y_minibatch, self.learning_rate/T::from(batch_size).unwrap(), &mut cc);
            }

            let result = self.evaluate(test.x, test.y, test.size);
            println!("Epoch {}: accuracy {:?} {}/{} {} (MiniBatch)", e, result.0, result.1, result.2, MINIBATCH_EPOCHS);
        }
    }

    pub fn train_iteration(&mut self, x_iteration: ArrayView2<T>, y_iteration: &Array2<T>, learning_rate: T, cc: &mut CacheComputation<T>) {
        self.forward_pass(x_iteration, cc);
        let chain_rule_compute = self.backward_pass(y_iteration, cc);
        self.update_iteration(learning_rate, chain_rule_compute);
    }

    // forward pass is a wrapper around predict as it tracks the intermediate linear and non-linear values
    pub fn forward_pass(&self, x: ArrayView2<T>, cc: &mut CacheComputation<T>) {
        cc.init(x.to_owned());
        let mut opt_comp = Some(cc);
        self.predict(x, &mut opt_comp);
    }

    pub fn predict(&self, x: ArrayView2<T>, opt: &mut Option<&mut CacheComputation<T>>) -> Array2<T> {
        let mut z: Array2<T>;
        let mut a: Array2<T>;
        let mut acc = x.to_owned();

        // Compute and store the linear Z values and nonlinear A (activation) values
        // Z = W*A0 + B, A1 = RELU(Z) or A2 = Sigmoid(Z)
        for ((w, b), a_func) in self.weights.iter().zip(self.biases.iter()).zip(self.forward.iter()) {
            z = acc.weighted_sum(w, b); // linear, z = w.dot(&acc) + b
            a = z.activate(a_func); // non-linear, σ(z)

            acc = a;
            opt.as_mut().map(|cc| cc.cache(z, &acc));
        }

        acc // return last computed activation values
    }

    pub fn backward_pass<'b, 'c>(&self, y: &Array2<T>, cc: &'b mut CacheComputation<T>) -> ChainRuleComputation<'c, T>
    where 'b: 'c // cc is around longer than crc
    {
        // Compute the chain rule values for each layer
        // Store the partial cost derivative for biases and weights from each layer,
        // starting with last layer first

        let mut crc = ChainRuleComputation::new(cc);
        let acc0: Array2<T> = crc.init(y);

        // zip number of iterations with corresponding weight (start from back to front layer)
        let zipped = (0..self.total_layers-2).zip(self.weights.iter().rev());
        zipped.fold(acc0, |acc: Array2<T>, (_, w)| {
            crc.fold_layer(acc, w)
        });

        crc
    }

    pub fn update_iteration(&mut self, learning_rate: T, crc: ChainRuleComputation<T>) {
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

    pub fn evaluate(&self, x_test: &Array2<T>, y_test: &Array2<T>, n_test: usize) -> (f64, usize, usize) {
        // run forward pass with no caching of intermediate values on each observation data
        let mut output: Array2<T>;
        let mut empty: Option<&mut CacheComputation<T>> = None;
        let mut matches: usize = 0;

        // processes an x_test row of input values at a time
        for (x_sample, y) in x_test.axis_chunks_iter(Axis(0), 1).zip(y_test.iter()) {
            output = self.predict(x_sample.t(), &mut empty);

//            println!("predict x_sample {} out {} am {} y {:?}", x_sample.view(), &output.view(), arg_max(&output), y);
            if output.arg_max() == T::to_usize(y).unwrap() {
                matches += 1;
            }
        }

        let accuracy = matches as f64 / n_test as f64;

        (accuracy, matches, n_test)
    }
}
