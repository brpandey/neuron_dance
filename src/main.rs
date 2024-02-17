use simple_network::{
    network::Network,
    activation::Activation,
    dataset::DataSet
};

fn main() {
    let path = "./data.csv";

    // train / test ratio
    let train_size = 2.0/3.0;

    let ds = DataSet::new(path);
    let tts_data = ds.train_test_split(train_size);

    // construct activation function trait objects from supplied strs
    let layers = vec!["relu", "sigmoid"]
        .into_iter()
        .map(|l| l.parse().unwrap())
        .collect::<Vec<Box<dyn Activation<f64>>>>();

    let mut simple_nn = Network::new(vec![3, 3, 1], layers, 0.01);
    simple_nn.train_sgd(tts_data.get_ref());
    //simple_nn.train_minibatch(tts_data.get_ref(), 32);
}
