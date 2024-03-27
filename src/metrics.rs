use std::{fmt, collections::HashMap};
use ndarray::{Array2, arr2};

use crate::cost::CostFp;
use crate::types::Batch;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]

pub enum Mett { // Metrics Type
    Accuracy,
    Cost,
}

#[derive(Debug)]
pub struct Metrics {
    metrics_map: HashMap<Mett, bool>,
    cost_fp: CostFp,
    one_hot: HashMap<usize, Array2<f64>>,
}

impl Metrics {
    pub fn new(metrics_list: Vec<Mett>, cost_fp: CostFp, output_size: usize) -> Self {
        // reduce metrics list into hash table for easy lookup
        let metrics_map: HashMap<Mett, bool> = metrics_list.into_iter().map(|v| (v, true)).collect();
        let mut one_hot = HashMap::new();
        let mut zeros: Array2<f64>;

        for i in 0..output_size {
            zeros = Array2::zeros((output_size, 1));
            zeros[(i,0)] = 1.;
            one_hot.insert(i, zeros);
        }

        Self { metrics_map, cost_fp, one_hot }
    }

    pub fn create_tally(&mut self, batch_type: Option<Batch>,
                    epoch: (usize, usize)) -> Tally {
        Tally::new(
            self.metrics_map.clone(), self.cost_fp.clone(),
            batch_type, epoch, self.one_hot.clone(),
        )
    }
}

pub struct Tally {
    metrics_map: HashMap<Mett, bool>,
    cost_fp: CostFp,
    batch_type: Option<Batch>,
    epoch: (usize, usize), // current epoch, total epochs
    one_hot: HashMap<usize, Array2<f64>>,
    total_cost: f64,
    total_matches: usize,
    total_size: usize,
    accuracy: Option<AccuracyMetric>,
    loss: Option<LossMetric>,
}

impl Tally {
    pub fn new(metrics_map: HashMap<Mett, bool>, cost_fp: CostFp, batch_type: Option<Batch>,
               epoch: (usize, usize), one_hot: HashMap<usize, Array2<f64>>) -> Self {
        Self {
            metrics_map,
            cost_fp,
            batch_type,
            epoch,
            one_hot,
            total_cost: 0.0,
            total_matches: 0,
            total_size: 0,
            accuracy: None, // haven't computed yet
            loss: None, // haven't computed yet
        }
    }

    pub fn t_match(&mut self) { // tally or track matches
        if self.metrics_map.contains_key(&Mett::Accuracy) {
            self.total_matches += 1;
        }
    }

    pub fn t_cost(&mut self, a: &Array2<f64>, y: usize) { // tally or track cost
        if self.metrics_map.contains_key(&Mett::Cost) {
            let v_y: &Array2<f64>;
            let temp;
            if a.shape()[0] == 1 && a.shape()[1] == 1 {
                temp = arr2(&[[y as f64]]);
                v_y = &temp;
            } else {
                v_y = self.one_hot(y).unwrap();
            }
//            println!("t_cost: a output is {:?}, y is {:?}, vectored y is {:?}", &a, y, &v_y);

            let cost = (self.cost_fp)(a, v_y);
            self.total_cost += cost;
//            println!("total_cost is {:?},  cost is {:?}", self.total_cost, cost);
        }
    }

    pub fn summarize(&mut self, n_total: usize) {
        self.total_size = if n_total >= 1 { n_total } else { panic!("total size can't be zero") };

        if self.metrics_map.contains_key(&Mett::Accuracy) {
            self.accuracy = Some(AccuracyMetric::new(self.total_matches, self.total_size));
            self.total_matches = 0; // reset
        }

        if self.metrics_map.contains_key(&Mett::Cost) {
            self.loss = Some(LossMetric::new(self.total_cost, self.total_size));
            self.total_cost = 0.0; // reset
        }
    }

    pub fn display(&self) {
        // generate text related to batch type
        let b = self.batch_type.as_ref().map_or(String::from(""), |v| v.to_string());

        // generate initial accuracy and loss texts
        let mut a_txt = self.accuracy.as_ref().map(|v| format!("{} {}", v, b));
        let mut l_txt = self.loss.as_ref().map(|v| format!("{} {}", v, b));

        // if necessary (if minibatch), prefix initial texts with epoch info
        if let Some(&Batch::Mini(_)) = self.batch_type.as_ref() {
            let e_txt = format!("Epoch {}/{}:", self.epoch.0, self.epoch.1);
            a_txt = self.accuracy.as_ref().map(|_| format!("{} {}", &e_txt, a_txt.unwrap()));
            l_txt = self.loss.as_ref().map(|_| format!("{} {}", &e_txt, l_txt.unwrap()));
        }

        // print metrics related display text
        a_txt.as_ref().map(|v| println!("{v}"));
        l_txt.as_ref().map(|v| println!("{v}"));
    }

    pub fn one_hot(&self, index: usize) -> Option<&Array2<f64>> { self.one_hot.get(&index) }
}


struct AccuracyMetric(f64, usize, usize);

impl AccuracyMetric {
    pub fn new(matches: usize, n_total: usize) -> Self {
        if n_total == 0 { panic!("total size can't be zero"); }
        AccuracyMetric(matches as f64/n_total as f64, matches, n_total)
    }
}

impl fmt::Display for AccuracyMetric {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Accuracy {:.4} {}/{}", self.0, self.1, self.2)
    }
}

struct LossMetric(f64, f64, usize);

impl LossMetric {
    pub fn new(total_cost: f64, n_total: usize) -> Self {
        if n_total == 0 { panic!("total size can't be zero"); }
        LossMetric(total_cost/n_total as f64, total_cost, n_total)
    }
}

impl fmt::Display for LossMetric {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Avg Loss {:.4} {:.4}/{}", self.0, self.1, self.2)
    }
}
