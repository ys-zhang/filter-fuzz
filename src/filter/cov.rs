use super::{tf, utils, Debug, Filter};
use libafl::{
    inputs::HasBytesVec,
    observers::{map::MapObserver, ObserversTuple},
};
use num_traits::{cast::cast, PrimInt};
use std::{marker::PhantomData, vec};

struct CovFilterStat {
    num_in: usize,         // number of inputs fed in
    num_out: usize,        // number of inputs that gets out of the filter
    num_obv: usize,        // number of real execution
    hits: Vec<f32>,        // sum of observed coverage map
    hits_normed: Vec<f32>, // standerized hits
    sim_smpls: Vec<f32>,   // history of similarity value observed
}

impl CovFilterStat {
    fn new(cov_map_len: usize, window_size: usize) -> CovFilterStat {
        Self {
            num_in: 0,
            num_out: 0,
            num_obv: 0,
            hits: vec![0.0; cov_map_len],
            hits_normed: vec![0.0; cov_map_len],
            sim_smpls: Vec::with_capacity(window_size),
        }
    }

    fn update_std_hits(&mut self, _ys: &[&[f32]]) {
        todo!("yun")
    }
}

#[allow(unused)]
pub struct CovFilter<I, O, T>
where
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    /// name used get the MapObserver from ObserversTuple
    name: String,
    batch_size: usize,

    stat: CovFilterStat,

    obv_xs: Vec<Vec<u8>>,
    obv_ys: Vec<Vec<f32>>,

    model: utils::Model,
    phantom: PhantomData<(I, O, T)>,
}

impl<I, O, T> CovFilter<I, O, T>
where
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    pub fn new(name: &str, batch_size: usize) -> Self {
        let model = utils::Model::new(name);
        let obv_xs = Vec::with_capacity(batch_size);
        let obv_ys = Vec::with_capacity(batch_size);
        Self {
            name: name.to_string(),
            batch_size,
            stat: CovFilterStat::new(model.out_dim),

            obv_xs,
            obv_ys,

            model,
            phantom: PhantomData,
        }
    }
}

impl<I, O, T> CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    fn filter(&mut self, ys: &tf::Tensor<f32>) -> Vec<f32> {
        // (num_of_sample, sample_dim)
        let (n, m) = {
            let shape = &ys.shape();
            (shape[0].unwrap() as usize, shape[1].unwrap() as usize)
        };

        self.stat.num_in += n;

        let prob = {
            let similarities: Vec<_> = ys
                .windows(m)
                .map(|y| Self::similarity(y, &self.stat.hits_normed))
                .collect();

            similarities
        };
        prob
    }

    /// calc similarity btw standardized coverage maps
    /// the two coverage maps must have same length
    #[inline]
    fn similarity(std_cv_mp1: &[f32], std_cv_mp2: &[f32]) -> f32 {
        std_cv_mp1
            .iter()
            .zip(std_cv_mp2.iter())
            .map(|(s, t)| s * t)
            .sum()
    }
}

impl<I, O, S, T> Filter<I, S> for CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn run(&mut self, batch: Vec<I>, _state: &mut S, _corpus_id: usize) -> (Vec<I>, Vec<f32>) {
        let xs: Vec<&[u8]> = batch.iter().map(|x| x.bytes()).collect();
        let ys_normed = self.model.predict_normed(&xs); // `ys` is normalized
        let prob = self.filter(&ys_normed);
        // self.update_hits(&ys);
        (batch, prob)
    }

    fn observe<OT: ObserversTuple<I, S>>(&mut self, observers: &OT, input: &I) {
        let observer = observers.match_name::<O>(&self.name).unwrap();
        let map = observer.map().unwrap();

        // observe the sample
        self.stat.num_obv += 1;
        self.obv_xs.push(input.bytes().to_vec());
        self.obv_ys
            .push(map.iter().map(|&t| cast(t).unwrap()).collect());

        // try train the model
        if self.obv_xs.len() == self.batch_size {
            let xs: Vec<_> = self.obv_xs.iter().map(|x| x.as_slice()).collect();
            let ys: Vec<_> = self.obv_ys.iter().map(|y| y.as_slice()).collect();

            self.stat.update_std_hits(&ys);

            self.model.train(&xs, &ys, 32); // TODO: hardcode, train for 32 steps
            self.obv_xs.truncate(0);
            self.obv_ys.truncate(0);
        }
    }
}
