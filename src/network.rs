/// Network
/// Represents a neural network model with its weight and bias connections
/// In totality it represents an idealized function sculpted by its training data
/// Through successive approximations the model minimizes the respective target or loss function
/// increasing its accuracy and utility in predicting new unseen data

extern crate blas_src; // C & Fortran linear algebra library for optimized matrix compute

use std::iter::Iterator;
use ndarray::{Array2, ArrayView2, Axis};
use rand::{Rng, seq::SliceRandom};

use crate::{activation::ActFp, algebra::AlgebraExt}; // import local traits
use crate::gradient_cache::{GradientCache, GT};
use crate::hypers::Hypers;
use crate::chain_rule::ChainRuleComputation;
use crate::dataset::TrainTestSubsetRef;
use crate::layers::{Layer, LayerStack, LayerTerms};
use crate::cost::{Cost, Loss};
use crate::metrics::{Metrics, Tally};
use crate::types::{Batch, Eval, Metr};
use crate::optimizer::{Optim, ParamKey};

#[derive(Debug)]
pub struct Network {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    forward: Vec<ActFp>, // forward propagation, activation functions
    layers: Option<LayerStack>,
    hypers: Hypers,
    cache: Option<GradientCache>,
    metrics: Option<Metrics>,
}

impl Network {
    pub fn new() -> Self { Self::empty() }

    fn empty() -> Self {
        Network {
            weights: vec![], biases: vec![], forward: vec![], 
            layers: Some(LayerStack::new()), hypers: Hypers::empty(),
            cache: None, metrics: None,
        }
    }

    /**** Public associated methods ****/

    pub fn add<L: Layer<Output = LayerTerms> + 'static>(&mut self, layer: L) {
        self.layers.as_mut().unwrap().add(layer);
    }

    pub fn compile<'a>(&mut self, loss_type: Loss, learning_rate: f64, l2_rate: f64, metrics_type: Metr<'a>) {
        let (output_size, weights, biases, forward, backward, output_act) = self.layers.as_mut().unwrap().reduce();

        let loss: Box<dyn Cost> = loss_type.into();
        let (fp_cost, fp_cost_deriv, fp_cost_comb_rule) = loss.triple();

        let metrics = Some(Metrics::new(metrics_type, fp_cost, output_size, l2_rate));
        let hypers = Hypers::new(learning_rate, l2_rate);

        // initialize network properly
        let n = Network {
            weights, biases, forward,
            layers: self.layers.take(),
            cache: None, hypers, metrics,
        };

        let _ = std::mem::replace(self, n); // replace empty network with new initialized network

        let gc = GradientCache::new(
            backward, &self.biases, output_size,
            (fp_cost_deriv, fp_cost_comb_rule),
            output_act
        );

        self.cache = Some(gc);
    }

    /// Train model with relevant dataset given the specified hyperparameters
    pub fn fit(&mut self, subsets: &TrainTestSubsetRef, epochs: usize, batch_type: Batch, eval: Eval) {
        let mut cache = self.cache.take().unwrap();
        let mut optt = Optim::Default;

        self.hypers.set_batch_type(batch_type);
        cache.set_batch_type(batch_type);

        match batch_type {
            Batch::SGD => self.train_sgd(subsets, epochs, &mut cache, eval),
            Batch::Mini(_) => (),
            Batch::Mini_(_, o) => optt = o,
        }

        if batch_type.is_mini() {
            self.hypers.set_optimizer(optt.into(), optt);
            self.train_minibatch(subsets, epochs, batch_type.value(), &mut cache, eval)
        }
    }

    pub fn eval(&mut self, subsets: &TrainTestSubsetRef, eval: Eval) {
        let mut tally = self.metrics.as_mut().unwrap().create_tally(None, (0, 0));
        self.evaluate(subsets, &eval, &mut tally);
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let mut none = None;
        self.predict_(x, &mut none)
    }

    pub fn predict_random(&self, subsets: &TrainTestSubsetRef, eval: Eval) -> usize {
        use crate::visualize::Visualize;
        use crate::pool::Pool;

        let mut none = None;

        let subset_ref = match eval {
            Eval::Train => subsets.0, // train subset
            Eval::Test => subsets.1, // test subset
        };

        let (x, y) = subset_ref.random();
        let output = self.predict_(x.t(), &mut none);

        let y_pred = output.arg_max();
        let y_label = y[(0,0)] as usize;

        if y_label == y_pred {
            println!("Successful y prediction, correct label is {y_label} \n -- See reduced, randomly chosen x input image below -- ");
        } else {
            println!("No match! y prediction {y_pred} is different from correct y label {y_label} \n -- See reduced, randomly chosen x input image below --");
        }

        let image = x.into_shape((28, 28)).unwrap();
        let reduced_image = Pool::apply(image.view(), 2, 2);
        Visualize::table_preview(&reduced_image.view(), None, true);

        y_label
    }

    /**** Private associated methods ****/

    /// Sgd employs 1 sample chosen at random from the data set for gradient descent
    fn train_sgd(&mut self, subsets: &TrainTestSubsetRef, epochs: usize, gc: &mut GradientCache, eval: Eval) {
        let (mut rng, mut random_index);
        let (mut x_single, mut y_single);
        let train = subsets.0;

        rng = rand::thread_rng();
        for _ in 0..epochs { // SGD_EPOCHS { // train and update network based on single observation sample
            random_index = rng.gen_range(0..train.size);
            x_single = train.x.select(Axis(0), &[random_index]); // arr2(&[[0.93333333, 0.93333333, 0.81960784]])
            y_single = train.y.select(Axis(0), &[random_index]); // arr2(&[[1.0]])

            self.train_iteration(
                x_single.t(), &y_single,
                gc, self.hypers.learning_rate(),
                self.hypers.l2_regularization_rate(), train.size, 0,
            );
        }

        let (batch, epoch) = (Some(Batch::SGD), (0, epochs));
        let mut tally = self.metrics.as_mut().unwrap().create_tally(batch, epoch);
        self.evaluate(subsets, &eval, &mut tally);
    }

    /// Minibatch uses batch_size samples chosen at random for gradient descent
    /// until it has covered all of the data set to form a single epoch out of a handful of epochs.

    /// Note: Probably more accurate to call it train stochastic mini batch
    fn train_minibatch(&mut self, subsets: &TrainTestSubsetRef, epochs: usize,
                       batch_size: usize, gc: &mut GradientCache, eval: Eval) {

        let (mut x_minibatch, mut y_minibatch);
        let optimizer_type = self.hypers.optimizer_type();
        let train = subsets.0;
        let mut tally;

        let mut row_indices = (0..train.size).collect::<Vec<usize>>();
        let b = Some(Batch::Mini_(batch_size, optimizer_type));
        let mut r = rand::thread_rng();

        for e in 0..epochs {
            row_indices.shuffle(&mut r);

            for c in row_indices.chunks(batch_size) { //train and update network after each batch size of observation samples
                x_minibatch = train.x.select(Axis(0), &c);
                y_minibatch = train.y.select(Axis(0), &c);

                // transpose to ensure proper matrix multi fit
                self.train_iteration(
                    x_minibatch.t(), &y_minibatch,
                    gc, self.hypers.learning_rate(),
                    self.hypers.l2_regularization_rate(), train.size, e,
                );
            }

            tally = self.metrics.as_mut().unwrap().create_tally(b, (e, epochs));
            self.evaluate(subsets, &eval, &mut tally);
        }
    }

    fn train_iteration(&mut self, x_iteration: ArrayView2<f64>, y_iteration: &Array2<f64>,
                       gc: &mut GradientCache, lr: f64, l2_rate: f64, total_size: usize, t: usize) {

        gc.add(GT::Features, x_iteration.to_owned());
        self.forward_pass(x_iteration, gc);
        let chain_rule_compute = self.backward_pass(y_iteration, gc);
        self.update_iteration(chain_rule_compute, lr, l2_rate, total_size, t);
    }

    // forward pass is a wrapper around predict as it tracks the intermediate linear and non-linear values
    fn forward_pass(&self, x: ArrayView2<f64>, gc: &mut GradientCache) {
        let mut wrapped = Some(gc);
        self.predict_(x, &mut wrapped);
    }

    /// After model has been trained, apply new data on the model and its inherent
    /// parameters (weights, biases) to generate the output classes
    fn predict_(&self, x: ArrayView2<f64>, wrapped: &mut Option<&mut GradientCache>) -> Array2<f64> {
        let mut z: Array2<f64>;
        let mut a: Array2<f64>;
        let mut acc = x.to_owned();

        // An artificial neutron contains both a linear function (weighted_sum) with parameters
        // and a nonlinear activation function (act_fun)

        // Compute and store the linear Z values and nonlinear A (activation) values
        // Z = W*A0 + B, A1 = RELU(Z) or A2 = Sigmoid(Z)

        for ((w, b), act_fun) in self.weights.iter().zip(self.biases.iter()).zip(self.forward.iter()) {
            z = acc.weighted_sum(w, b); // linear, z = w.dot(&acc) + b
            a = (act_fun)(&z); // non-linear,  σ(z)

            wrapped.as_mut().map(|c| {
                c.add(GT::Linear, z);
                c.add(GT::Nonlinear, (&a).to_owned());
            });

            acc = a;
        }

        acc // return last computed activation values
    }

    fn backward_pass<'a, 'b>(&self, y: &Array2<f64>, gc: &'a mut GradientCache) -> ChainRuleComputation<'b>
    where 'a: 'b // tc is around longer than crc
    {
        // Compute the chain rule values for each layer
        // Store the partial cost derivative for biases and weights from each layer,
        // starting with last layer first

        let total_layers = self.layers.as_ref().unwrap().len();
        let mut crc = ChainRuleComputation::new(gc);
        let acc0: Array2<f64> = crc.init(y);

        // zip number of iterations with corresponding weight (start from back to front layer)
        let zipped = (0..total_layers-2).zip(self.weights.iter().rev());
        zipped.fold(acc0, |acc: Array2<f64>, (_, w)| {
            crc.fold_layer(acc, w)
        });

        crc
    }

    fn update_iteration(&mut self, crc: ChainRuleComputation, learning_rate: f64, l2_rate: f64, n_total: usize, t: usize) {
        // Apply delta contributions to current biases and weights by subtracting
        // The negative gradient uses the chain rule to find a local
        // minima in our neural network cost graph as opposed to maxima (positive gradient)
        let (mut adj, mut velocity, mut key);
        let optimizer = self.hypers.optimizer();

        let (b_deltas, w_deltas) = (crc.bias_deltas(), crc.weight_deltas());

        for (i, (b, db)) in self.biases.iter_mut().zip(b_deltas).enumerate() {
            // optimizer, if enabled, is used to calibrate the constant learning rate
            // more accurately with more data
            key = ParamKey::BiasGradient(i as u8);
            adj = optimizer.calculate(key, &db, t);
            velocity = adj.mapv(|x| x * learning_rate);

            *b -= &velocity;
        }

        //  weight_decay factor is 1−ηλ/n
        let weight_decay = 1.0-learning_rate*(l2_rate/n_total as f64);

        for (i, (w, dw)) in self.weights.iter_mut().zip(w_deltas).enumerate() {
            // optimizer, if enabled, is used to calibrate the constant learning rate
            // more accurately with more data
            key = ParamKey::WeightGradient(i as u8);
            adj = optimizer.calculate(key, &dw, t);
            velocity = adj.mapv(|x| x * learning_rate);

            *w = &*w*weight_decay - velocity;
        }
    }

    /// Compare the classes represented by the prediction output with the known classes
    /// represented by the data set's test y data. Upon match, increase the model's accuracy
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
        let mut none = None;
        let mut label_index;

        // processes an x_test row of input values at a time
        for (x_sample, y) in x_data.axis_chunks_iter(Axis(0), 1).zip(y_data.iter()) {
            output = self.predict_(x_sample.t(), &mut none);

            label_index = *y as usize;

            tally.t_cost(&output, label_index);

            if output.arg_max() == label_index {
                tally.t_match();
            }
        }

        for w in self.weights.iter() {
            tally.regularize_cost(w, n_data);
        }

        tally.summarize(n_data);
        tally.display();
    }
}
