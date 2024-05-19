use clap::{Arg, ArgAction, Command};
use neuron_dance::{
    dataset::{
        csv::{CSVData, CSVType},
        idx::{MnistData, MnistType},
        DataSet, TrainTestSubsetRef,
    },
    layers::{Act, Batch, Dense, Eval, Input1, Input2, Loss, Metr, Optim, Weit},
    network::Network,
};

fn main() {
    let mut matches = Command::new("neuron_dance")
        .about("Neuron Dance")
        .arg(
            Arg::new("type")
                .action(ArgAction::Set)
                .value_parser(["csv1", "diab", "iris", "mnist", "fash", "pre"])
                .default_value("csv1")
                .help("Specify network type")
                .short('t')
                .long("type")
                .value_name("NETWORK TYPE"),
        )
        .get_matches();

    let ntype: NetworkType = matches
        .remove_one::<String>("type")
        .unwrap()
        .parse()
        .unwrap();

    let token: String = ntype.to_string();

    let train_percentage = 2.0 / 3.0; // train / total ratio, test = total - train
    let mut dataset: Box<dyn DataSet>;

    dataset = match ntype {
        NetworkType::CSV1 => Box::new(CSVData::new(CSVType::RGB)),
        NetworkType::Diabetes => Box::new(CSVData::new(CSVType::Custom("diabetes", None))),
        NetworkType::Iris => Box::new(CSVData::new(CSVType::Custom("iris", Some(vec!["Setosa", "Virginica", "Versicolor"])))),
        NetworkType::Mnist => Box::new(MnistData::new(MnistType::Regular)),
        NetworkType::FashionMnist | NetworkType::Preload => Box::new(MnistData::new(MnistType::Fashion)),
    };

    if let Err(e) = dataset.fetch() { println!("{}", &e); std::process::exit(1) };
    dataset.shuffle();
    dataset.head();

    let mut tts = dataset.train_test_split(train_percentage);
    let mut subsets = tts.get_ref();
    let mut model;

    match ntype {
        NetworkType::CSV1 => {
            model = Network::new();
            model.add(Input1(3));
            model.add(Dense(3, Act::Relu));
            model.add(Dense(1, Act::Sigmoid));
            model.compile(Loss::Quadratic, 0.2, 0.0, Metr(" accuracy , cost"));
            model.fit(&subsets, 10000, Batch::SGD, Eval::Train); // using SGD approach (doesn't have momentum supported)
        },
        NetworkType::Diabetes => {
            tts = tts.min_max_scale(0.0, 1.0); // scale down the features to a 0..1 scale for better model performance
            subsets = tts.get_ref();

            model = Network::new();
            model.add(Input1(8));
            model.add(Dense(12, Act::Relu));
            model.add(Dense(8, Act::Relu));
            model.add(Dense(1, Act::Sigmoid_(Weit::GlorotN)));
            model.compile(Loss::BinaryCrossEntropy, 0.5, 0.0, Metr("accuracy, cost"));
            model.fit(&subsets, 50, Batch::Mini(10), Eval::Train);
            model.view();
        },
        NetworkType::Iris => {
            model = Network::new();
            model.add(Input1(4));
            model.add(Dense(10, Act::Relu));
            model.add(Dense(10, Act::Relu));
            model.add(Dense(3, Act::Sigmoid));
            model.compile(Loss::BinaryCrossEntropy, 0.005, 0.3, Metr("Accuracy, cost"));
            model.fit(&subsets, 50, Batch::Mini(5), Eval::Test);

            // Now that model has been trained, make random selections
            random_predicts(&model, &subsets);
        },
        NetworkType::Mnist => {
            // Layers near input learn more basic qualities of the dataset thus bigger size
            model = Network::new();
            model.add(Input2(28, 28));
            model.add(Dense(100, Act::Sigmoid_(Weit::GlorotN)));
            model.add(Dense(10, Act::Sigmoid_(Weit::GlorotN))); // Layers near output learn more advanced qualities
            model.compile(Loss::BinaryCrossEntropy, 0.1, 5.0, Metr("accuracy"));
            model.fit(&subsets, 3, Batch::Mini_(10, Optim::Adam), Eval::Test);

            random_predicts(&model, &subsets); // Now that model has been trained, make random selections
            //            model.view();
        },
        NetworkType::FashionMnist => {
            // Layers near input learn more basic qualities of the dataset thus bigger size
            model = Network::new();
            model.add(Input2(28, 28));
            model.add(Dense(128, Act::Relu));
            model.add(Dense(10, Act::Softmax_(Weit::GlorotN))); // Layers near output learn more advanced qualities
            model.compile(Loss::CategoricalCrossEntropy, 0.1, 5.0, Metr("accuracy"));
            model.fit(&subsets, 10, Batch::Mini_(5, Optim::Default), Eval::Test);
        },
        NetworkType::Preload => {
            let tok = NetworkType::FashionMnist.to_string();
            model = Network::load(&tok);

            random_predicts(&model, &subsets); // Now that model has been trained, make random selections
        }
    }

    model.eval(&subsets, Eval::Test);

    if ntype != NetworkType::Preload {
        model.store(&token);

        let mut newmodel = Network::load(&token);
        newmodel.eval(&subsets, Eval::Test);
    }
}

pub fn random_predicts<'a>(model: &Network, subsets: &TrainTestSubsetRef<'a>) {
    // make random selections for 4 individual images from either Test or Train set
    model.predict_using_random(&subsets, Eval::Test);
    model.predict_using_random(&subsets, Eval::Train);
    model.predict_using_random(&subsets, Eval::Test);
    model.predict_using_random(&subsets, Eval::Train);
}

#[derive(PartialEq, strum_macros::Display, strum_macros::EnumString)]
#[strum(serialize_all = "lowercase")]
pub enum NetworkType {
    CSV1,
    #[strum(serialize = "diab")]
    Diabetes,
    Iris,
    Mnist,
    #[strum(serialize = "fash")]
    FashionMnist,
    #[strum(serialize = "pre")]
    Preload,
}
