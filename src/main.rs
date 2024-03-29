use simple_network::{
    network::Network,
    dataset::{DataSet, csv::{CSVType, CSVData},
              idx::{MnistType, MnistData}},
    layers::{Act, Batch, Eval, Loss, Metr, Input1, Input2, Dense},
};

pub enum NetworkType {
    CSV1,
    CSV2,
    Mnist,
}

fn main() {
    let ntype = NetworkType::Mnist;
    let train_percentage = 2.0/3.0;     // train / total ratio, test = total - train
    let mut dataset: Box<dyn DataSet>;

    dataset = match ntype {
        NetworkType::CSV1 => Box::new(CSVData::new(CSVType::RGB)),
        NetworkType::CSV2 => Box::new(CSVData::new(CSVType::Custom("diabetes"))),
        NetworkType::Mnist => Box::new(MnistData::new(MnistType::Regular)),
    };

    let tts = dataset.train_test_split(train_percentage);
    let mut model;
    let subsets = tts.get_ref();

    match ntype {
        NetworkType::CSV1 => {
            model = Network::new();
            model.add(Input1(3));
            model.add(Dense(3, Act::Relu));
            model.add(Dense(1, Act::Sigmoid));
            model.compile(Loss::Quadratic, 0.2, Metr(" accuracy , cost"));
            model.fit(&subsets, 10000, Batch::SGD, Eval::Train); // using SGD approach (doesn't have momentum supported)
        },
        NetworkType::CSV2 => {
            model = Network::new();
            model.add(Input1(8));
            model.add(Dense(12, Act::Relu));
            model.add(Dense(8, Act::Relu));
            model.add(Dense(2, Act::Sigmoid));
            model.compile(Loss::CrossEntropy, 0.5, Metr(" accuracy"));
            model.fit(&subsets, 10, Batch::Mini(20), Eval::Test); // using SGD approach (doesn't have momentum supported)
        },
        NetworkType::Mnist => {
            model = Network::new();
            model.add(Input2(28, 28));
            model.add(Dense(30, Act::Sigmoid));
            model.add(Dense(10, Act::Sigmoid));
            model.compile(Loss::CrossEntropy, 0.5, Metr("accuracy,cost "));
            model.fit(&subsets, 4, Batch::Mini(32), Eval::Test);
        }
    }
    model.eval(&subsets, Eval::Test);
}
