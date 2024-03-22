extern crate blas_src; // C & Fortran linear algebra library for optimized matrix compute

use std::iter::Iterator;
use ndarray::{Array, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use rand::{Rng, seq::SliceRandom};
use statrs::distribution::Normal;

use crate::{activation::ActFp, algebra::AlgebraExt}; // import local traits
use crate::term_cache::TermCache;
use crate::chain_rule::ChainRuleComputation;
use crate::dataset::TrainTestSubsetRef;
use crate::layers::{Layer, LayerStack, LayerTerms};
use crate::cost::{Cost, functions::Loss};
use crate::metrics::{Mett, Metrics, Tally};
use crate::types::{Batch, Eval};

#[derive(Debug)]
pub struct Network {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    forward: Vec<ActFp>, // forward propagation, activation functions
    layers: Option<LayerStack>,
    cache: Option<TermCache>,
    metrics: Option<Metrics>,
}

impl Network {
    pub fn new() -> Self { Self::empty() }

    fn empty() -> Self {
        Network {
            weights: vec![], biases: vec![], forward: vec![], 
            layers: Some(LayerStack::new()), cache: None, metrics: None,
        }
    }

    /**** Public member functions ****/

    pub fn add<L: Layer<Output = LayerTerms> + 'static>(&mut self, layer: L) {
        self.layers.as_mut().unwrap().add(layer);
    }

    pub fn compile(&mut self, loss_type: Loss, learning_rate: f64, metrics_type: Vec<Mett>) {
        let (sizes, forward, backward) = self.layers.as_mut().unwrap().reduce();
        let total_layers = self.layers.as_ref().unwrap().len();

        let loss: Box<dyn Cost> = loss_type.into();
        let (cost_fp, cost_deriv_fp) = loss.pair();

        let metrics = Some(Metrics::new(metrics_type, cost_fp));

        let (mut weights, mut biases): (Vec<Array2<f64>>, Vec<Array2<f64>>) = (vec![], vec![]);
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

        // initialize network properly
        let n = Network {
            weights, biases, forward,
            layers: self.layers.take(), cache: None, metrics,
        };

        let _ = std::mem::replace(self, n); // replace empty network with new initialized network

        self.cache = Some(TermCache::new(backward, &self.biases, sizes[total_layers-1],
                                         learning_rate, Batch::SGD, cost_deriv_fp));
    }

    pub fn fit(&mut self, subsets: &TrainTestSubsetRef, epochs: usize, batch_type: Batch, eval: Eval) {
        let mut cache = self.cache.take().unwrap();
        cache.set_batch_type(batch_type);

        match batch_type {
            Batch::SGD => self.train_sgd(subsets, epochs, &mut cache, eval),
            Batch::Mini(batch_size) => self.train_minibatch(subsets, epochs, batch_size, &mut cache, eval),
        }
    }

    pub fn eval(&mut self, subsets: &TrainTestSubsetRef, eval: Eval) {
        let mut tally = self.metrics.as_mut().unwrap().create_tally(None, (0, 0));
        self.evaluate(subsets, &eval, &mut tally);
    }

     /**** Private member functions ****/

    fn train_sgd(&mut self, subsets: &TrainTestSubsetRef, epochs: usize, tc: &mut TermCache, eval: Eval) {
        let (mut rng, mut random_index);
        let (mut x_single, mut y_single);
        let train = subsets.0;

        for _ in 0..epochs { // SGD_EPOCHS { // train and update network based on single observation sample
            rng = rand::thread_rng();
            random_index = rng.gen_range(0..train.size);
            x_single = train.x.select(Axis(0), &[random_index]); // arr2(&[[0.93333333, 0.93333333, 0.81960784]])
            y_single = train.y.select(Axis(0), &[random_index]); // arr2(&[[1.0]])

            self.train_iteration(
                x_single.t(), &y_single,
                tc, tc.learning_rate()
            );
        }

        let (batch, epoch) = (Some(Batch::SGD), (0, epochs));
        let mut tally = self.metrics.as_mut().unwrap().create_tally(batch, epoch);
        self.evaluate(subsets, &eval, &mut tally);
    }

    fn train_minibatch(&mut self, subsets: &TrainTestSubsetRef, epochs: usize,
                       batch_size: usize, tc: &mut TermCache, eval: Eval) {
        let (mut x_minibatch, mut y_minibatch);
        let train = subsets.0;
        let mut row_indices = (0..train.size).collect::<Vec<usize>>();
        let b = Some(Batch::Mini(batch_size));
        let mut tally;

        for e in 0..epochs {
            row_indices.shuffle(&mut rand::thread_rng());

            for c in row_indices.chunks(batch_size) { //train and update network after each batch size of observation samples
                x_minibatch = train.x.select(Axis(0), &c);
                y_minibatch = train.y.select(Axis(0), &c);

                // transpose to ensure proper matrix multi fit
                self.train_iteration(
                    x_minibatch.t(), &y_minibatch,
                    tc, tc.learning_rate()
                );
            }

            tally = self.metrics.as_mut().unwrap().create_tally(b, (e, epochs));
            self.evaluate(subsets, &eval, &mut tally);
        }
    }

    fn train_iteration(&mut self, x_iteration: ArrayView2<f64>, y_iteration: &Array2<f64>, cache: &mut TermCache, lr: f64) {
        self.forward_pass(x_iteration, cache);
        let chain_rule_compute = self.backward_pass(y_iteration, cache);
        self.update_iteration(chain_rule_compute, lr);
    }

    // forward pass is a wrapper around predict as it tracks the intermediate linear and non-linear values
    fn forward_pass(&self, x: ArrayView2<f64>, tc: &mut TermCache) {
        tc.stack.reset(x.to_owned());
        let mut opt = Some(tc);
        self.predict(x, &mut opt);
    }

    fn predict(&self, x: ArrayView2<f64>, opt: &mut Option<&mut TermCache>) -> Array2<f64> {
        let mut z: Array2<f64>;
        let mut a: Array2<f64>;
        let mut acc = x.to_owned();

        // Compute and store the linear Z values and nonlinear A (activation) values
        // Z = W*A0 + B, A1 = RELU(Z) or A2 = Sigmoid(Z)
        for ((w, b), a_func) in self.weights.iter().zip(self.biases.iter()).zip(self.forward.iter()) {
            z = acc.weighted_sum(w, b); // linear, z = w.dot(&acc) + b
            a = z.activate(a_func); // non-linear, σ(z)

            acc = a;
            opt.as_mut().map(|c| c.stack.push(z, &acc));
        }

        acc // return last computed activation values
    }

    fn backward_pass<'b, 'c>(&self, y: &Array2<f64>, tc: &'b mut TermCache) -> ChainRuleComputation<'c>
    where 'b: 'c // tc is around longer than crc
    {
        // Compute the chain rule values for each layer
        // Store the partial cost derivative for biases and weights from each layer,
        // starting with last layer first

        let total_layers = self.layers.as_ref().unwrap().len();
        let mut crc = ChainRuleComputation::new(tc);
        let acc0: Array2<f64> = crc.init(y);

        // zip number of iterations with corresponding weight (start from back to front layer)
        let zipped = (0..total_layers-2).zip(self.weights.iter().rev());
        zipped.fold(acc0, |acc: Array2<f64>, (_, w)| {
            crc.fold_layer(acc, w)
        });

        crc
    }

    fn update_iteration(&mut self, crc: ChainRuleComputation, learning_rate: f64) {
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

    fn evaluate(&self, subsets: &TrainTestSubsetRef, eval: &Eval,
                tally: &mut Tally) {

        let s = match *eval {
            Eval::Train => subsets.0, // train subset
            Eval::Test => subsets.1, // test subset
        };

        // retrieve eval data, labels, and size
        let (x_data, y_data, n_data) : (&Array2<f64>, &Array2<f64>, usize) = (s.x, s.y, s.size);

        // run forward pass with no caching of intermediate values on each observation data
        let mut output: Array2<f64>;
        let mut empty: Option<&mut TermCache> = None;

        // processes an x_test row of input values at a time
        for (x_sample, y) in x_data.axis_chunks_iter(Axis(0), 1).zip(y_data.iter()) {
            output = self.predict(x_sample.t(), &mut empty);

            if output.arg_max() == *y as usize {
                tally.t_match();
            }
        }

        tally.summarize(n_data);
        tally.display();
    }
}
