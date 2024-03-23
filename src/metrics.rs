use std::collections::HashMap;
use ndarray::Array2;

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
    accuracy: Option<f64>,
    avg_loss: Option<f64>,
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
            avg_loss: None, // haven't computed yet
        }
    }

    pub fn t_match(&mut self) { // tally or track matches
        if self.metrics_map.contains_key(&Mett::Accuracy) {
            self.total_matches += 1;
        }
    }

    //    pub fn t_cost(&mut self, a: &Array2<f64>, y: &Array2<f64>) { // tally or track cost
    pub fn t_cost(&mut self, a: &Array2<f64>, y: usize) { // tally or track cost
        if self.metrics_map.contains_key(&Mett::Cost) {
            let v_y = self.one_hot(y).unwrap();

//            println!("t_cost: a output is {:?}, y is {:?}, vectored y is {:?}", &a, y, &v_y);

            let cost = (self.cost_fp)(a, v_y);
            self.total_cost += cost;
//            println!("total_cost is {:?},  cost is {:?}", self.total_cost, cost);
        }
    }

    pub fn summarize(&mut self, n_total: usize) {
        self.total_size = if n_total >= 1 { n_total } else { panic!("total size can't be zero") };

        if self.metrics_map.contains_key(&Mett::Accuracy) {
            self.accuracy = Some(self.total_matches as f64 / self.total_size as f64);
        }

        if self.metrics_map.contains_key(&Mett::Cost) {
            self.avg_loss = Some(self.total_cost / self.total_size as f64);
        }
    }

    pub fn display(&self) {
        // No batch type
        if self.batch_type.is_none() {
            if self.accuracy.is_some() {
                println!("Accuracy {:.4} {}/{}", self.accuracy.as_ref().unwrap(),
                         self.total_matches, self.total_size);
            }

            if self.avg_loss.is_some() {
                println!("Avg Loss {:.4} {:.4}/{}", self.avg_loss.as_ref().unwrap(),
                         self.total_cost, self.total_size);
            }

            return
        }

        match self.batch_type.as_ref().unwrap() {
            Batch::SGD => {
                if self.accuracy.is_some() {
                    println!("Accuracy {:.4} {}/{} {} (SGD)", self.accuracy.as_ref().unwrap(),
                             self.total_matches, self.total_size, self.epoch.1);
                }

                if self.avg_loss.is_some() {
                    println!("Avg Loss {:.4} {:.4}/{} {} (SGD)", self.avg_loss.as_ref().unwrap(),
                             self.total_cost, self.total_size, self.epoch.1);
                }

            },
            Batch::Mini(_) => {
                if self.accuracy.is_some() {
                    println!("Epoch {}: accuracy {:.4} {}/{} {} (MiniBatch)", self.epoch.0,
                             self.accuracy.as_ref().unwrap(), self.total_matches, self.total_size, self.epoch.1);
                }

                if self.avg_loss.is_some() {
                    println!("Epoch {}: avg loss {:.4} {:.4}/{} {} (MiniBatch)", self.epoch.0,
                             self.avg_loss.as_ref().unwrap(), self.total_cost, self.total_size, self.epoch.1);
                }

            }
        }
    }

    pub fn one_hot(&self, index: usize) -> Option<&Array2<f64>> { self.one_hot.get(&index) }
}
