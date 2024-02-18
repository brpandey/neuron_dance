use simple_network::{
    network::Network,
    dataset::DataSet
};

fn main() {
    let path = "./data.csv";
    let train_size = 2.0/3.0; // train / total ratio, test ratio would be total - train/ total
    let learning_rate = 0.01_f32; // float by default is f64, so for f32 append _f32
    // this param is used for Network type parameter

    let ds = DataSet::new(path);
    let tts_data = ds.train_test_split(train_size);

    let mut model = Network::new(vec![3, 3, 1], vec!["relu", "sigmoid"], learning_rate);
    model.train_sgd(tts_data.get_ref());
    //model.train_minibatch(tts_data.get_ref(), 32);
}
