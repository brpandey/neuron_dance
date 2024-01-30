use simple_network::{
    network::Network,
    activation::functions::{relu::Relu, sigmoid::Sigmoid},
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

    let mut simple_nn = Network::new(vec![3, 3, 1], vec![Box::new(Relu), Box::new(Sigmoid)], 0.01);
    simple_nn.train_sgd(&x_train, &y_train, n_train, &x_test, &y_test, n_test);
}
