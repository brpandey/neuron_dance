use simple_network::{
    network::Network,
    activation::Activation,
    dataset::{DataSet, csv::{CSV, CSVData},
              idx::{Mnist, MnistData}},
};

fn main() {
    let tts_data;
    let csv = false;
    let train_size = 2.0/3.0;     // train / total ratio, test = total - train

    let dataset: Box<dyn DataSet>;

    if csv {
        dataset = Box::new(CSVData::new(CSV::RGB));
    } else {
        dataset = Box::new(MnistData::new(Mnist::Regular));
    }

    tts_data = dataset.train_test_split(train_size);

    // construct activation function trait objects from supplied strs
    let layers = vec!["sigmoid", "sigmoid"]
        .into_iter()
        .map(|l| l.parse().unwrap())
        .collect::<Vec<Box<dyn Activation>>>();

    let mut model;

    if csv {
        model = Network::new(vec![3, 3, 1], layers, 0.2);
    } else {
        let sizes = vec![784, 50, 10];
        model = Network::new(sizes, layers, 0.3);
    }

    //model.train_sgd(tts_data.get_ref());
    model.train_minibatch(tts_data.get_ref(), 32);
}
