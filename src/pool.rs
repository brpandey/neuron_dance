use ndarray::{s, Array, Array2, ArrayView2};
use ndarray_stats::QuantileExt;

pub struct Pool;

impl Pool {
    pub fn apply(a: ArrayView2<f64>, pool_size: usize, stride_size: usize) -> Array2<f64> {
        let mut vec = vec![];

        let side = a.shape()[0];

        // extract features
        // generate convolutional kernel (filter) matrixes - unweighted
        let mut kernel;
        for r in (0..side).step_by(stride_size) {
            for c in (0..side).step_by(stride_size) {
                kernel = a.slice(s![r..(r+pool_size), c..(c+pool_size)]);
                vec.push(kernel);
            }
        }

        // downsample feature maps
        // create max pool, taking the max value from each kernel matrix
        let max_pool: Vec<f64> = vec.iter().map(|f| {
            *f.max().unwrap()
        }).collect();

        // max pool is (14,14) as max len is 196 side len is 14
        let len = max_pool.len();
        let side = num::integer::sqrt(len);

        Array::from_shape_vec((side, side), max_pool).unwrap()
    }
}
