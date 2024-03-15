use simple_network::{
    network::Network,
    dataset::{DataSet, csv::{CSVType, CSVData},
              idx::{MnistType, MnistData}},
    layers::{Input1, Input2, Dense},
};

fn main() {
    let tts_data;
    let csv = false;
    let train_size = 2.0/3.0;     // train / total ratio, test = total - train

    let mut dataset: Box<dyn DataSet>;

    if csv {
        dataset = Box::new(CSVData::new(CSVType::RGB));
    } else {
        dataset = Box::new(MnistData::new(MnistType::Regular));
    }

    tts_data = dataset.train_test_split(train_size);

    let mut model;

    if csv {
        model = Network::new();
        model.add(Input1(3));
        model.add(Dense(3, "relu".to_string()));
        model.add(Dense(1, "sigmoid".to_string()));
    } else {
        model = Network::new();
        model.add(Input2(28, 28));
        model.add(Dense(50, "sigmoid".to_string()));
        model.add(Dense(10, "sigmoid".to_string()));
    }

    //model.compile("quadratic_cost", "adam", 0.2, "loss, accuracy");
    model.compile(0.2);

    // model.fit(tts_data.get_ref(), 20, 32); // 20 epochs, 32 batch size

    //model.train_sgd(tts_data.get_ref());
    model.train_minibatch(tts_data.get_ref(), 32);

    //model.evaluate()
}
