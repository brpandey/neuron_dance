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
    let ((x_train, y_train), (x_test, y_test), (n_train, n_test)) = ds.train_test_split(train_size);

//    println!("train X  {:?} -------- train Y {:?}", &x_train, &y_train);
//    println!("test X {:?} --------- test Y {:?}", &x_test, &y_test);

    // construct activation function trait objects from supplied strs
    let layers = vec!["relu", "sigmoid"].into_iter().map(|l| l.parse().unwrap()).collect::<Vec<Box<dyn Activation>>>();

    let mut simple_nn = Network::new(vec![3, 3, 1], layers, 0.01);
    simple_nn.train_sgd(&x_train, &y_train, n_train, &x_test, &y_test, n_test);
}
