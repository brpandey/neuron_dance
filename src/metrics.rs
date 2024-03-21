use std::collections::HashMap;
use ndarray::Array2;

use crate::cost::CostFp;
use crate::network::Batch;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Mett { // Metrics Type
    Accuracy,
    Cost,
}

#[derive(Debug)]
pub struct Metrics {
    metrics_map: HashMap<Mett, bool>,
    cost_fp: CostFp,
}

impl Metrics {
    pub fn new(metrics_list: Vec<Mett>, cost_fp: CostFp) -> Self {
        // reduce metrics list into hash table for easy lookup
        let metrics_map: HashMap<Mett, bool> = metrics_list.into_iter().map(|v| (v, true)).collect();

        Self {
            metrics_map,
            cost_fp,
        }
    }

    pub fn recorder(&mut self, batch_type: Option<Batch>,
                    epoch: (usize, usize)) -> MetricsRecorder {

        MetricsRecorder::new(
            self.metrics_map.clone(), self.cost_fp.clone(),
            batch_type, epoch
        )
    }
}

pub struct MetricsRecorder {
    metrics_map: HashMap<Mett, bool>,
    cost_fp: CostFp,
    batch_type: Option<Batch>,
    epoch: (usize, usize), // current epoch, total epochs
    total_cost: f64,
    total_matches: usize,
    total_size: usize,
    accuracy: Option<f64>,
    avg_loss: Option<f64>,
}

impl MetricsRecorder {
    pub fn new(metrics_map: HashMap<Mett, bool>, cost_fp: CostFp, batch_type: Option<Batch>,
               epoch: (usize, usize)) -> Self {

        Self {
            metrics_map,
            cost_fp,
            batch_type,
            epoch,
            total_cost: 0.0,
            total_matches: 0,
            total_size: 0,
            accuracy: None, // haven't computed yet
            avg_loss: None, // haven't computed yet
        }
    }

    pub fn record_match(&mut self) {
        if self.metrics_map.contains_key(&Mett::Accuracy) {
            self.total_matches += 1;
        }
    }

    pub fn record_cost(&mut self, a: &Array2<f64>, y: &Array2<f64>) {
        if self.metrics_map.contains_key(&Mett::Cost) {
            let cost = (self.cost_fp)(a, y);
            self.total_cost += cost;
            println!("total_cost is {:?},  cost is {:?}", self.total_cost, cost);
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
                println!("Accuracy {} {}/{}", self.accuracy.as_ref().unwrap(),
                         self.total_matches, self.total_size);
            }

            if self.avg_loss.is_some() {
                println!("Avg Loss {} {}/{}", self.avg_loss.as_ref().unwrap(),
                         self.total_cost, self.total_size);
            }

            return
        }

        match self.batch_type.as_ref().unwrap() {
            Batch::SGD => {
                if self.accuracy.is_some() {
                    println!("Accuracy {} {}/{} {} (SGD)", self.accuracy.as_ref().unwrap(),
                             self.total_matches, self.total_size, self.epoch.1);
                }

                if self.avg_loss.is_some() {
                    println!("Avg Loss {} {}/{} {} (SGD)", self.avg_loss.as_ref().unwrap(),
                             self.total_cost, self.total_size, self.epoch.1);
                }

            },
            Batch::Mini(_) => {
                if self.accuracy.is_some() {
                    println!("Epoch {}: accuracy {:?} {}/{} {} (MiniBatch)", self.epoch.0,
                             self.accuracy.as_ref().unwrap(), self.total_matches, self.total_size, self.epoch.1);
                }

                if self.avg_loss.is_some() {
                    println!("Epoch {}: avg loss {:?} {}/{} {} (MiniBatch)", self.epoch.0,
                             self.avg_loss.as_ref().unwrap(), self.total_cost, self.total_size, self.epoch.1);
                }

            }
        }
    }
}
