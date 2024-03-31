use clap::{Arg, ArgAction, Command};
use std::str::FromStr;
use simple_network::{
    network::Network,
    dataset::{DataSet, csv::{CSVType, CSVData},
              idx::{MnistType, MnistData}},
    layers::{Act, Batch, Eval, Loss, Metr, Input1, Input2, Dense},
};

fn main() {
    let mut matches = Command::new("simple_network")
        .about("Simple Neural Network")
        .arg(
            Arg::new("type")
                .action(ArgAction::Set)
                .value_parser(["csv1", "csv2", "mnist"])
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
            model.compile(Loss::Quadratic, 0.2, 0.0, Metr(" accuracy , cost"));
            model.fit(&subsets, 10000, Batch::SGD, Eval::Train); // using SGD approach (doesn't have momentum supported)
        },
        NetworkType::CSV2 => {
            model = Network::new();
            model.add(Input1(8));
            model.add(Dense(8, Act::Relu));
            model.add(Dense(3, Act::Relu));
            model.add(Dense(1, Act::Sigmoid));
            model.compile(Loss::CrossEntropy, 0.5, 0.3, Metr("accuracy, cost"));
            model.fit(&subsets, 100000, Batch::SGD, Eval::Train);
//            model.fit(&subsets, 100, Batch::Mini(20), Eval::Test); // using SGD approach (doesn't have momentum supported)
        },
        NetworkType::Mnist => {
            model = Network::new();
            model.add(Input2(28, 28));
            model.add(Dense(100, Act::Sigmoid));
            model.add(Dense(10, Act::Sigmoid));
            model.compile(Loss::CrossEntropy, 0.1, 5.0, Metr("accuracy,cost "));
            model.fit(&subsets, 30, Batch::Mini(10), Eval::Test);
        }
    }
    model.eval(&subsets, Eval::Test);
}

pub enum NetworkType {
    CSV1,
    CSV2,
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
            "mnist" => Ok(NetworkType::Mnist),
            _ => Err(NTParseError),
        }
    }
}
