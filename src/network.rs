/// Network
/// Represents a neural network model with its weight and bias connections
/// In totality it represents an idealized function sculpted by its training data
/// Through successive approximations the model minimizes the respective target or loss function
/// increasing its accuracy and utility in predicting new unseen data
extern crate blas_src; // C & Fortran linear algebra library for optimized matrix compute
use ndarray::{Array2, ArrayView2, Axis};
use rand::{seq::SliceRandom, Rng};
use std::{default::Default, iter::Iterator, ops::Add};

use crate::{
    activation::ActFp,
    algebra::AlgebraExt, // import local traits
    archive::NetworkArchive,
    chain_rule::ChainRuleComputation,
    cost::{Cost, Loss},
    dataset::TrainTestSubsets,
    gradient_cache::{GradientCache, GT},
    hypers::Hypers,
    layers::{Layer, LayerStack, LayerTerms},
    metrics::{Metrics, Tally},
    optimizer::{Optim, ParamKey},
    save::Save,
    types::{Batch, Eval, Metr, ModelState, SimpleError},
};

const WRONG_ORDER: &str = "User error ~ wrong order of operations";

#[derive(Debug, Default)]
pub struct Network {
    layers: Option<LayerStack>,
    current_state: ModelState,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    forward: Vec<ActFp>, // forward propagation, activation functions
    hypers: Hypers,
    cache: Option<GradientCache>,
    metrics: Option<Metrics>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            layers: Some(LayerStack::new()),
            ..Default::default()
        }
    }

    /**** Public associated methods ****/

    pub fn add<L: Layer<Output = LayerTerms> + 'static>(&mut self, layer: L) {
        self.check_valid_state(ModelState::Add).expect(WRONG_ORDER);
        self.layers.as_mut().unwrap().add(layer);
        self.current_state = ModelState::Add;
    }

    pub fn compile(
        &mut self,
        loss_type: Loss,
        learning_rate: f64,
        l2_rate: f64,
        metrics_type: Metr<'_>,
    ) {
        self.check_valid_state(ModelState::Compile)
            .expect(WRONG_ORDER);

        let (size_ends, weights, biases, forward, backward, acts) =
            self.layers.as_mut().unwrap().reduce();
        let output_act = *acts.last().unwrap();
        let (input_size, output_size) = size_ends;

        let loss: Box<dyn Cost> = loss_type.into();
        let (cost_fp, cost_deriv_fp, cost_comb_rule_fp) = loss.triple();

        let metrics = Some(Metrics::new(metrics_type, cost_fp, output_size, l2_rate));
        let hypers = Hypers::new(
            learning_rate,
            l2_rate,
            input_size,
            output_size,
            acts,
            loss_type,
        );

        // initialize network properly
        let n = Network {
            weights,
            biases,
            forward,
            layers: self.layers.take(),
            cache: None,
            hypers,
            metrics,
            current_state: ModelState::Compile,
        };

        let _ = std::mem::replace(self, n); // replace empty network with new initialized network

        let gc = GradientCache::new(
            backward,
            &self.biases,
            output_size,
            (cost_deriv_fp, cost_comb_rule_fp),
            output_act,
        );

        self.cache = Some(gc);
    }

    /// Train model with relevant dataset given the specified hyperparameters
    pub fn fit(
        &mut self,
        subsets: &TrainTestSubsets,
        epochs: usize,
        batch_type: Batch,
        eval: Eval,
    ) -> Result<(), SimpleError> {
        self.check_valid_state(ModelState::Fit).expect(WRONG_ORDER);
        let (layer_size, n_features) = (self.hypers.input_size(), subsets.num_features());

        if layer_size != n_features {
            // input layer size specified incorrectly
            let e = SimpleError::InputLayerSizeNoMatch(layer_size, n_features);
            eprintln!("Unrecoverable error: {}", &e);
            return Err(e);
        }

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
            self.hypers.set_optimizer(optt);
            self.train_minibatch(subsets, epochs, batch_type.value(), &mut cache, eval)
        }

        self.current_state = ModelState::Fit;

        Ok(())
    }

    pub fn eval(&mut self, subsets: &TrainTestSubsets, eval: Eval) {
        self.check_valid_state(ModelState::Eval).expect(WRONG_ORDER);
        let mut tally = self.metrics.as_mut().unwrap().create_tally(None, (0, 0));
        self.evaluate(subsets, &eval, &mut tally);
    }

    pub fn predict(&self, x: ArrayView2<f64>) {
        self.check_valid_state(ModelState::Eval).expect(WRONG_ORDER);
        let mut none = None;
        self.predict_(x, &mut none);
    }

    pub fn predict_using_random(&self, subsets: &TrainTestSubsets, eval: Eval) -> usize {
        self.check_valid_state(ModelState::Eval).expect(WRONG_ORDER);
        let mut none = None;

        let subset_ref = subsets.subset_ref(&eval);
        let (x, y) = subset_ref.random(); // retrieve random set of input and output pairs
        let output = self.predict_(x.t(), &mut none);

        let class_names: Option<&Vec<String>> = subset_ref.class_names();

        let y_pred = output.arg_max();

        // map y values to a class names value if one is provided
        let y_pred_txt = class_names.map_or(y_pred.to_string(), |vec| {
            vec.get(y_pred).map_or("?".to_string(), |v| v.clone())
        });

        let y_label = y[(0, 0)] as usize;

        let y_label_txt = class_names.map_or(y_label.to_string(), |vec| {
            vec.get(y_label).map_or("?".to_string(), |v| v.clone())
        });

        let s_txt = format!("[Successful y prediction] correct label is {y_pred_txt}");
        let f_txt = format!(
            "[No match!] y prediction {y_pred_txt} is different from correct y label {y_label_txt}"
        );

        if y_label == y_pred {
            println!("{s_txt}")
        } else {
            println!("{f_txt}")
        }

        subset_ref.features_peek(&x);

        y_label
    }

    pub fn store(&mut self, token: &str) -> Result<(), SimpleError> {
        if self.current_state < ModelState::Fit {
            panic!("Unable to store model since it hasn't been properly fitted");
        } // only store fitted models

        let filename1 = format!("{}-network-dump.txt", token);
        let filename2 = format!("{}-hypers-dump.txt", token);

        self.save(filename1)?;
        self.hypers.save(filename2)?;

        Ok(())
    }

    pub fn load(token: &str) -> Result<Self, SimpleError> {
        let filename1 = format!("{}-network-dump.txt", token);
        let filename2 = format!("{}-hypers-dump.txt", token);

        let net = <Network as Save>::restore(filename1)?;
        let hypers = <Hypers as Save>::restore(filename2)?;

        Ok(net + hypers)
    }

    pub fn view(&self) {
        println!("{:#?}", &self);
    }

    /**** Pub crate associated methods ****/

    pub(crate) fn weights(&self) -> &Vec<Array2<f64>> {
        self.weights.as_ref()
    }
    pub(crate) fn biases(&self) -> &Vec<Array2<f64>> {
        self.biases.as_ref()
    }

    /**** Private associated methods ****/

    /// Sgd employs 1 sample chosen at random from the data set for gradient descent
    fn train_sgd(
        &mut self,
        subsets: &TrainTestSubsets,
        epochs: usize,
        gc: &mut GradientCache,
        eval: Eval,
    ) {
        let (mut rng, mut random_index);
        let (mut x_single, mut y_single);
        let train = subsets.train();

        rng = rand::thread_rng();
        for _ in 0..epochs {
            // SGD_EPOCHS { // train and update network based on single observation sample
            random_index = rng.gen_range(0..train.size);
            x_single = train.x.select(Axis(0), &[random_index]); // arr2(&[[0.93333333, 0.93333333, 0.81960784]])
            y_single = train.y.select(Axis(0), &[random_index]); // arr2(&[[1.0]])

            self.train_iteration(x_single.t(), &y_single, gc, train.size, 0);
        }

        let (batch, epoch) = (Some(Batch::SGD), (0, epochs));
        let mut tally = self.metrics.as_mut().unwrap().create_tally(batch, epoch);
        self.evaluate(subsets, &eval, &mut tally);
    }

    /// Minibatch uses batch_size samples chosen at random for gradient descent
    /// until it has covered all of the data set to form a single epoch out of a handful of epochs.

    /// Note: Probably more accurate to call it train stochastic mini batch
    fn train_minibatch(
        &mut self,
        subsets: &TrainTestSubsets,
        epochs: usize,
        batch_size: usize,
        gc: &mut GradientCache,
        eval: Eval,
    ) {
        let (mut x_minibatch, mut y_minibatch);
        let optimizer_type = self.hypers.optimizer_type();
        let train = subsets.train();
        let mut tally;

        let mut row_indices = (0..train.size).collect::<Vec<usize>>();
        let b = Some(Batch::Mini_(batch_size, optimizer_type));
        let mut r = rand::thread_rng();

        for e in 0..epochs {
            row_indices.shuffle(&mut r);

            for chunk_ref in row_indices.chunks(batch_size) {
                //train and update network after each batch size of observation samples
                x_minibatch = train.x.select(Axis(0), chunk_ref);
                y_minibatch = train.y.select(Axis(0), chunk_ref);

                // transpose to ensure proper matrix multi fit
                self.train_iteration(x_minibatch.t(), &y_minibatch, gc, train.size, e);
            }

            tally = self
                .metrics
                .as_mut()
                .unwrap()
                .create_tally(b, (e + 1, epochs));
            self.evaluate(subsets, &eval, &mut tally);
        }
    }

    // Note - Consider grouping parameters into subtype?
    fn train_iteration(
        &mut self,
        x_iteration: ArrayView2<f64>,
        y_iteration: &Array2<f64>,
        gc: &mut GradientCache,
        total_size: usize,
        t: usize,
    ) {
        gc.add(GT::Features, x_iteration.to_owned());
        self.forward_pass(x_iteration, gc);
        let chain_rule_compute = self.backward_pass(y_iteration, gc);
        self.update_iteration(chain_rule_compute, total_size, t);
    }

    // forward pass is a wrapper around predict as it tracks the intermediate linear and non-linear values
    fn forward_pass(&self, x: ArrayView2<f64>, gc: &mut GradientCache) {
        let mut wrapped = Some(gc);
        self.predict_(x, &mut wrapped);
    }

    /// After model has been trained, apply new data on the model and its inherent
    /// parameters (weights, biases) to generate the output classes
    fn predict_(
        &self,
        x: ArrayView2<f64>,
        wrapped: &mut Option<&mut GradientCache>,
    ) -> Array2<f64> {
        let mut z: Array2<f64>;
        let mut a: Array2<f64>;
        let mut acc = x.to_owned();

        // An artificial neutron contains both a linear function (weighted_sum) with parameters
        // and a nonlinear activation function (act_fun)

        // Compute and store the linear Z values and nonlinear A (activation) values
        // Z = W*A0 + B, A1 = RELU(Z) or A2 = Sigmoid(Z)

        for ((w, b), act_fun) in self
            .weights
            .iter()
            .zip(self.biases.iter())
            .zip(self.forward.iter())
        {
            z = acc.weighted_sum(w, b); // linear, z = w.dot(&acc) + b
            a = (act_fun)(&z); // non-linear,  σ(z)

            if let Some(cache) = wrapped.as_mut() {
                cache.add(GT::Linear, z);
                cache.add(GT::Nonlinear, a.to_owned());
            }

            acc = a;
        }

        acc // return last computed activation values
    }

    fn backward_pass<'a, 'b>(
        &self,
        y: &Array2<f64>,
        gc: &'a mut GradientCache,
    ) -> ChainRuleComputation<'b>
    where
        'a: 'b, // tc is around longer than crc
    {
        // Compute the chain rule values for each layer
        // Store the partial cost derivative for biases and weights from each layer,
        // starting with last layer first

        let total_layers = self.layers.as_ref().unwrap().len();
        let mut crc = ChainRuleComputation::new(gc);
        let acc0: Array2<f64> = crc.init(y);

        // zip number of iterations with corresponding weight (start from back to front layer)
        let zipped = (0..total_layers - 2).zip(self.weights.iter().rev());
        zipped.fold(acc0, |acc: Array2<f64>, (_, w)| crc.fold_layer(acc, w));

        crc
    }

    fn update_iteration(&mut self, crc: ChainRuleComputation, n_total: usize, t: usize) {
        // Apply delta contributions to current biases and weights by subtracting
        // The negative gradient uses the chain rule to find a local
        // minima in our neural network cost graph as opposed to maxima (positive gradient)
        let (mut adj, mut velocity, mut key);

        let learning_rate = self.hypers.learning_rate();
        let l2_rate = self.hypers.l2_regularization_rate();

        let mut optimizer = self.hypers.optimizer();

        let (b_deltas, w_deltas) = (crc.bias_deltas(), crc.weight_deltas());

        for (i, (b, db)) in self.biases.iter_mut().zip(b_deltas).enumerate() {
            // optimizer, if enabled, is used to calibrate the constant learning rate
            // more accurately with more data
            key = ParamKey::BiasGradient(i as u8);
            adj = optimizer.calculate(key, db, t);
            velocity = adj.mapv(|x| x * learning_rate);

            *b -= &velocity;
        }

        //  weight_decay factor is 1−ηλ/n
        let weight_decay = 1.0 - learning_rate * (l2_rate / n_total as f64);

        for (i, (w, dw)) in self.weights.iter_mut().zip(w_deltas).enumerate() {
            // optimizer, if enabled, is used to calibrate the constant learning rate
            // more accurately with more data
            key = ParamKey::WeightGradient(i as u8);
            adj = optimizer.calculate(key, dw, t);
            velocity = adj.mapv(|x| x * learning_rate);

            *w = &*w * weight_decay - velocity;
        }
    }

    /// Compare the classes represented by the prediction output with the known classes
    /// represented by the data set's test y data. Upon match, increase the model's accuracy
    fn evaluate(&self, subsets: &TrainTestSubsets, eval: &Eval, tally: &mut Tally) {
        let s = subsets.subset_ref(eval);
        // retrieve eval data, labels, and size
        let (x_data, y_data, n_data): (&Array2<f64>, &Array2<f64>, usize) = (&s.x, &s.y, s.size);

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
        tally.display(true);
    }

    fn check_valid_state(&self, other: ModelState) -> Result<bool, SimpleError> {
        self.current_state.check_valid_state(&other)
    }
}

impl From<NetworkArchive> for Network {
    fn from(archive: NetworkArchive) -> Self {
        let mut a = archive;
        let (w_a, b_a) = (a.weights.take().unwrap(), a.biases.take().unwrap());

        Network {
            layers: None,
            current_state: ModelState::Fit, // set as a fitted model
            weights: w_a.into(),
            biases: b_a.into(),
            ..Default::default()
        }
    }
}

impl Save for Network {
    // select NetworkArcihve as intermediate structure
    type Proxy = NetworkArchive;
}

// Flesh out remaining network fields from hypers
impl Add<Hypers> for Network {
    type Output = Network;

    fn add(self, hypers: Hypers) -> Network {
        use crate::activation::ActivationFps;
        let mut network = self;

        let fps: ActivationFps = hypers.activations().into(); // convert Vec<Act> to Vec<ActFp>

        network.metrics = Some((&hypers).into()); // convert hypers into metrics
        network.hypers = hypers;
        network.forward = fps.into_inner();

        network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DataSet;
    use crate::layers::{Act, Dense, Input1};
    use crate::types::SimpleError;
    use std::sync::OnceLock;

    static TTS: OnceLock<TrainTestSubsets> = OnceLock::new();

    fn subsets_init() -> &'static TrainTestSubsets {
        use crate::dataset::csv::{CSVData, CSVType};

        TTS.get_or_init(|| {
            println!("Initializing subsets once (Should only see once)");
            let mut data = CSVData::new(CSVType::RGB);
            data.fetch().unwrap();
            data.train_test_split(2.0 / 3.0)
        })
    }

    #[test]
    fn model_check_compile_before_add() {
        // can't compile a model until layers have been added
        std::panic::set_hook(Box::new(|_| {})); // suppress panic output

        let result = std::panic::catch_unwind(|| {
            // compile before add
            let mut model = Network::new();
            model.compile(Loss::Quadratic, 0.1, 0.2, Metr("accuracy"));
        });

        assert!(&result.is_err());

        let result = std::panic::catch_unwind(|| {
            // compile after add
            let mut model = Network::new();
            Network::add(&mut model, Input1(3)); // qualified syntax for disambiguation
            Network::add(&mut model, Dense(3, Act::Relu));
            model.compile(Loss::Quadratic, 0.1, 0.2, Metr(" accuracy "));
        });

        assert!(&result.is_ok());
    }

    #[test]
    fn model_check_store_before_fit() {
        // can't store a model until it has been fitted
        std::panic::set_hook(Box::new(|_| {})); // suppress panic output

        // incorrect model construction pass
        let result = std::panic::catch_unwind(|| {
            let mut model = Network::new();
            Network::add(&mut model, Input1(3)); // qualified syntax for disambiguation
            Network::add(&mut model, Dense(3, Act::Relu));
            Network::add(&mut model, Dense(1, Act::Sigmoid));

            model.store("temp").unwrap();
        });

        assert!(&result.is_err());

        let err = result.unwrap_err();
        dbg!(err.downcast_ref::<&str>().unwrap());
        assert_eq!(
            *err.downcast_ref::<&str>().unwrap(),
            "Unable to store model since it hasn't been properly fitted"
        );

        let subsets = subsets_init();

        // correct model construction pass
        let result = std::panic::catch_unwind(|| {
            let mut model = Network::new();
            Network::add(&mut model, Input1(3)); // qualified syntax for disambiguation
            Network::add(&mut model, Dense(3, Act::Relu));
            Network::add(&mut model, Dense(1, Act::Sigmoid));
            model.compile(Loss::Quadratic, 0.2, 0.0, Metr(" "));
            model.fit(subsets, 1, Batch::SGD, Eval::Train).unwrap();
            model.store("temp").unwrap();
        });

        assert!(&result.is_ok());
    }

    #[test]
    fn model_load_before_store() {
        // can't store a model until it has been fitted
        let subsets = subsets_init();

        let mut model = Network::new();
        Network::add(&mut model, Input1(3));
        Network::add(&mut model, Dense(3, Act::Relu));
        Network::add(&mut model, Dense(1, Act::Sigmoid));
        model.compile(Loss::Quadratic, 0.2, 0.0, Metr(" "));
        model.fit(subsets, 1, Batch::SGD, Eval::Train).unwrap();

        // incorrect operation of loading before storing
        //        model.store("tempA").unwrap();
        let result = Network::load("tempA");

        assert!(&result.is_err());
        let error_str = &result.unwrap_err().to_string();
        assert_eq!(
            error_str,
            "Unable to perform IO -- No such file or directory (os error 2)"
        );
    }

    #[allow(unused_variables)]
    #[test]
    fn input_layer_dataset_shape_no_match() {
        let subsets = subsets_init();

        let n_features = subsets.num_features();
        let input_layer_size = 6;

        let mut model = Network::new();

        // input layer size is too big!
        Network::add(&mut model, Input1(input_layer_size));
        Network::add(&mut model, Dense(3, Act::Relu));
        Network::add(&mut model, Dense(1, Act::Sigmoid));
        model.compile(Loss::Quadratic, 0.2, 0.0, Metr(" "));

        let e = model.fit(subsets, 1, Batch::SGD, Eval::Train).unwrap_err();

        assert!(matches!(
            e,
            SimpleError::InputLayerSizeNoMatch(input_layer_size, n_features)
        ));
    }
}
