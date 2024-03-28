use std::{{fmt, fmt::Display}, collections::HashMap};
use ndarray::{Array2, arr2};

use crate::cost::CostFp;
use crate::types::{Batch, Metr, Mett};

#[derive(Debug)]
pub struct Metrics {
    metrics_map: HashMap<Mett, bool>,
    cost_fp: CostFp,
    one_hot: HashMap<usize, Array2<f64>>,
}

impl Metrics {
    pub fn new<'a>(mut metrics_type: Metr<'a>, cost_fp: CostFp, output_size: usize) -> Self {
        let metrics_list = metrics_type.to_vec();

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
    display_list: Vec<Box<dyn Display>>
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
            display_list: vec![],
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
        let total_size = if n_total >= 1 { n_total } else { panic!("total size can't be zero") };
        self.display_list = vec![]; // reset

        if self.metrics_map.contains_key(&Mett::Accuracy) {
            self.display_list.push(AccuracyM::new(self.total_matches, total_size));
            self.total_matches = 0; // reset
        }

        if self.metrics_map.contains_key(&Mett::Cost) {
            self.display_list.push(LossM::new(self.total_cost, total_size));
            self.total_cost = 0.0; // reset
        }
    }

    pub fn display(&self) {
        use std::collections::VecDeque;
        // generate text related to batch type leveraging Display trait
        let b = self.batch_type.as_ref().map_or(String::from(""), |v| v.to_string());

        // generate initial metrics texts, {custom metric text} {batch text}
        let mut m_txts: VecDeque<String> = self.display_list.iter()
            .map(|v| format!("{} {}", v, b)).collect();

        // if necessary (if minibatch), prefix initial texts with epoch info
        if let Some(&Batch::Mini(_)) = self.batch_type.as_ref() {
            let e = format!("Epoch {}/{}", self.epoch.0, self.epoch.1);
            m_txts.push_front(e);
            m_txts = m_txts.into_iter().enumerate().map(|(i,t)| {
                if i == 0 { t } else { format!("\t{}", t) }
            }).collect();
        }

        // print each specific metric's display text
        m_txts.iter().for_each(|t| println!("{t}"));
        println!("");
    }

    pub fn one_hot(&self, index: usize) -> Option<&Array2<f64>> { self.one_hot.get(&index) }
}


// define metric display structs
struct AccuracyM(f64, usize, usize);
struct LossM(f64, f64, usize);

impl AccuracyM {
    pub fn new(matches: usize, n_total: usize) -> Box<dyn Display> {
        if n_total == 0 { panic!("total size can't be zero"); }
        Box::new(AccuracyM(matches as f64/n_total as f64, matches, n_total))
    }
}

impl fmt::Display for AccuracyM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Accuracy {:.4} {}/{}", self.0, self.1, self.2)
    }
}

impl LossM {
    pub fn new(total_cost: f64, n_total: usize) -> Box<dyn Display> {
        if n_total == 0 { panic!("total size can't be zero"); }
        Box::new(LossM(total_cost/n_total as f64, total_cost, n_total))
    }
}

impl fmt::Display for LossM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Avg Loss {:.4} {:.4}/{}", self.0, self.1, self.2)
    }
}
