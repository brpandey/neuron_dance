use simple_network::{
    network::Network, activation::Act,
    dataset::{DataSet, csv::{CSVType, CSVData},
              idx::{MnistType, MnistData}},
    layers::{Input1, Input2, Dense},
};

fn main() {
    let csv = false;
    let train_percentage = 2.0/3.0;     // train / total ratio, test = total - train
    let mut dataset: Box<dyn DataSet>;

    if csv {
        dataset = Box::new(CSVData::new(CSVType::RGB));
    } else {
        dataset = Box::new(MnistData::new(MnistType::Regular));
    }

    let tts = dataset.train_test_split(train_percentage);
    let mut model;

    if csv {
        model = Network::new();
        model.add(Input1(3));
        model.add(Dense(3, Act::Relu));
        model.add(Dense(1, Act::Sigmoid));
    } else {
        model = Network::new();
        model.add(Input2(28, 28));
        model.add(Dense(50, Act::Sigmoid));
        model.add(Dense(10, Act::Sigmoid));
    }

    //model.compile("quadratic_cost", "adam", 0.2, "loss, accuracy");
    model.compile(0.3);

    // model.fit(tts.get_ref(), 20, 32); // 20 epochs, 32 batch size

    //model.train_sgd(tts.get_ref());
    model.train_minibatch(tts.get_ref(), 32);

    //model.evaluate()
}
