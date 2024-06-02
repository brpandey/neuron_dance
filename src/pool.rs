use ndarray::{s, Array, Array2, ArrayBase, Data, Ix2};
use ndarray_stats::QuantileExt;

pub enum PoolType {
    Max,
}

pub struct Pool;

impl Pool {
    pub fn apply<S>(
        a: &ArrayBase<S, Ix2>,
        pool_size: usize,
        stride_size: usize,
        pt: PoolType,
    ) -> Option<Array2<f64>>
    where
        S: Data<Elem = f64>,
    {
        let mut vec = vec![];
        let side = a.shape()[0];

        // extract features
        // generate convolutional kernel (filter) matrixes - unweighted
        let mut kernel;
        for r in (0..side).step_by(stride_size) {
            if r + pool_size > side {
                continue;
            } // don't overshoot side

            for c in (0..side).step_by(stride_size) {
                if c + pool_size > side {
                    continue;
                } // don't overshoot side

                kernel = a.slice(s![r..(r + pool_size), c..(c + pool_size)]);
                vec.push(kernel);
            }
        }

        // downsample feature maps
        let pool: Vec<f64> = vec
            .iter()
            .map(|f| {
                match pt {
                    // create max pool, taking the max value from each kernel matrix
                    PoolType::Max => *f.max().unwrap(),
                }
            })
            .collect();

        // if len is e.g. 196 pool is (14,14)
        let len = pool.len();
        let side = num::integer::sqrt(len);

        Array::from_shape_vec((side, side), pool).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    #[test]
    fn apply_max_pool() {
        let input: Array2<f64> = arr2(&[
            [4.0, 23.0, 7.0, 6.0],
            [5.0, 13.0, 4.0, 10.0],
            [19.0, 10.0, 6.0, 8.0],
            [2.0, 14.0, 5.0, 7.0],
        ]);

        let output2 = arr2(&[[23.0, 10.0], [19.0, 8.0]]);

        let output1 = arr2(&[[23.0, 23.0, 10.0], [19.0, 13.0, 10.0], [19.0, 14.0, 8.0]]);

        // stride 2
        let result = Pool::apply(&input, 2, 2, PoolType::Max).unwrap();

        assert_eq!(result, output2);

        // stride 1
        let result = Pool::apply(&input, 2, 1, PoolType::Max).unwrap();

        assert_eq!(result, output1);
    }
}
