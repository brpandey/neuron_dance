use clap::{Arg, ArgAction, Command};
use std::str::FromStr;
use simple_network::{
    network::Network,
    dataset::{DataSet, csv::{CSVType, CSVData},
              idx::{MnistType, MnistData}},
    layers::{Act, Batch, Eval, Loss, Metr, Input1, Input2, Dense, Weit, Optt},
};

fn main() {
    let mut matches = Command::new("simple_network")
        .about("Simple Neural Network")
        .arg(
            Arg::new("type")
                .action(ArgAction::Set)
                .value_parser(["csv1", "csv2", "iris", "mnist"])
                .default_value("csv1")
                .help("Specify network type")
                .short('t')
                .long("type")
                .value_name("NETWORK TYPE")
            )
        .get_matches();

    let ntype = matches.remove_one::<String>("type").unwrap().parse().unwrap();
    let train_percentage = 2.0/3.0;     // train / total ratio, test = total - train
    let mut dataset: Box<dyn DataSet>;

    dataset = match ntype {
        NetworkType::CSV1 => Box::new(CSVData::new(CSVType::RGB)),
        NetworkType::CSV2 => Box::new(CSVData::new(CSVType::Custom("diabetes"))),
        NetworkType::Iris => Box::new(CSVData::new(CSVType::Custom("iris"))),
        NetworkType::Mnist => Box::new(MnistData::new(MnistType::Regular)),
    };

    let mut tts = dataset.train_test_split(train_percentage);
    let mut subsets = tts.get_ref();
    let mut model;

    match ntype {
        NetworkType::CSV1 => {
            model = Network::new();
            model.add(Input1(3));
            model.add(Dense(3, Act::Relu));
            model.add(Dense(1, Act::Sigmoid));
            model.compile(Loss::Quadratic, 0.2, 0.0, Metr(" accuracy , cost"), Optt::Default);
            model.fit(&subsets, 10000, Batch::SGD, Eval::Train); // using SGD approach (doesn't have momentum supported)
        },
        NetworkType::CSV2 => {
            tts = tts.min_max_scale(0.0, 1.0); // scale down the features to a 0..1 scale for better model performance
            subsets = tts.get_ref();

            model = Network::new();
            model.add(Input1(8));
            model.add(Dense(12, Act::Relu));
            model.add(Dense(8, Act::Relu));
            model.add(Dense(1, Act::Sigmoid_(Weit::GlorotN)));
            model.compile(Loss::CrossEntropy, 0.5, 0.0, Metr("accuracy, cost"), Optt::Default);
            model.fit(&subsets, 120, Batch::Mini(10), Eval::Train);
        },
        NetworkType::Iris => {
            model = Network::new();
            model.add(Input1(4));
            model.add(Dense(10, Act::Relu));
            model.add(Dense(10, Act::Relu));
            model.add(Dense(3, Act::Sigmoid));
            model.compile(Loss::CrossEntropy, 0.005, 0.3, Metr("accuracy, cost"), Optt::Default);
            model.fit(&subsets, 100, Batch::Mini(5), Eval::Test);
        },
        NetworkType::Mnist => { // 784, 400, 400, 10
            model = Network::new();
            model.add(Input2(28, 28));
            model.add(Dense(100, Act::Sigmoid_(Weit::GlorotN)));
            model.add(Dense(10, Act::Sigmoid_(Weit::GlorotN)));
            model.compile(Loss::CrossEntropy, 0.1, 5.0, Metr("accuracy"), Optt::Adam);
            model.fit(&subsets, 20, Batch::Mini(32), Eval::Test);
        }
    }
    model.eval(&subsets, Eval::Test);
}

pub enum NetworkType {
    CSV1,
    CSV2,
    Iris,
    Mnist,
}

#[derive(Debug, PartialEq, Eq)]
pub struct NTParseError;

impl FromStr for NetworkType {
    type Err = NTParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "csv1" => Ok(NetworkType::CSV1),
            "csv2" => Ok(NetworkType::CSV2),
            "iris" => Ok(NetworkType::Iris),
            "mnist" => Ok(NetworkType::Mnist),
            _ => Err(NTParseError),
        }
    }
}
