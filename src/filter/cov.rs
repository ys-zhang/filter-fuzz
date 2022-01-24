use super::{tf, utils, Debug, Filter, FilterMode};
use libafl::{
    bolts::rands::{Rand, StdRand},
    inputs::HasBytesVec,
    observers::{map::MapObserver, ObserversTuple},
};
use num_traits::{cast::cast, PrimInt};
use std::{marker::PhantomData, vec};

// TODO: Hard code parameters
const RND_SEED: u64 = 8944; // random seed for filter
const TRN_STEP: usize = 32; // number of epoch for training
const WND_FACTOR: usize = 1 << 4; // filter stat's window size

struct CovFilterStat {
    num_in: usize,         // number of inputs fed in
    num_out: usize,        // number of inputs that gets out of the filter
    num_obv: usize,        // number of real execution
    hits: Vec<f32>,        // sum of observed coverage map
    hits_normed: Vec<f32>, // standerized hits
    smpl_wdw: Vec<f32>,    // sample window
    rand: StdRand,
}

impl CovFilterStat {
    fn new(cov_map_len: usize, window_size: usize) -> CovFilterStat {
        Self {
            num_in: 0,
            num_out: 0,
            num_obv: 0,
            hits: vec![0.0; cov_map_len],
            hits_normed: vec![0.0; cov_map_len],
            smpl_wdw: Vec::with_capacity(window_size),
            rand: StdRand::with_seed(RND_SEED),
        }
    }

    /// update cov map hits count, `ys` must have been compressed
    fn update_hits(&mut self, ys: &[&[f32]]) {
        // add new observed covmap to hits count
        for y in ys {
            for (s, t) in self.hits.iter_mut().zip(y.iter()) {
                *s += *t;
            }
        }
        // update the standardized hits count
        let mean = self.hits.iter().sum::<f32>() / (self.hits.len() as f32);
        let centered: Vec<_> = self.hits.iter().map(|&h| h - mean).collect();
        let norm = centered
            .iter()
            .map(|&h| (h - mean) * (h - mean))
            .sum::<f32>()
            .sqrt();
        for (h, ch) in self.hits_normed.iter_mut().zip(centered.iter()) {
            *h = *ch / norm;
        }
    }

    /// if sample window is full then randomly choose a sample from the window
    /// else return INFINITY
    fn rand_smpl(&mut self) -> f32 {
        if self.smpl_wdw.len() < self.smpl_wdw.capacity() {
            f32::INFINITY
        } else {
            *self.rand.choose(&self.smpl_wdw)
        }
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
    mode: FilterMode,
    stats: CovFilterStat,

    obv_xs: Vec<Vec<u8>>,  // buffer for inputs
    obv_ys: Vec<Vec<f32>>, // buffer for coverage map

    model: utils::Model,
    phantom: PhantomData<(I, O, T)>,
}

impl<I, O, T> CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    pub fn new(name: &str, batch_size: usize) -> Self {
        let model = utils::Model::new(name);
        let obv_xs = Vec::with_capacity(batch_size);
        let obv_ys = Vec::with_capacity(batch_size);
        let stats = CovFilterStat::new(model.out_dim, batch_size * WND_FACTOR);
        Self {
            name: name.to_string(),
            batch_size,
            mode: FilterMode::Preheat,
            stats,
            obv_xs,
            obv_ys,
            model,
            phantom: PhantomData,
        }
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

    /// compress a coverage map to fit the tensorflow model
    #[inline]
    fn compress(&mut self, map: &[T]) -> Vec<f32> {
        map.iter().map(|&t| cast(t).unwrap()).collect()
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

    fn filter(&mut self, batch: &[I], _state: &mut S, _corpus_idx: usize) -> Vec<bool> {
        let xs: Vec<&[u8]> = batch.iter().map(|x| x.bytes()).collect();
        let ys_normed = self.model.predict_normed(&xs); // `ys` is normalized
                                                        // (num_of_sample, sample_dim)
        let (n, m) = {
            let shape = &ys_normed.shape();
            (shape[0].unwrap() as usize, shape[1].unwrap() as usize)
        };

        let mut pass_num: usize = 0;

        let result = match self.mode {
            // in the preheat mode, fuzz all inputs
            FilterMode::Preheat => {
                pass_num = n;
                vec![true; pass_num]
            }
            FilterMode::Ready => ys_normed
                .windows(m)
                .map(|y| {
                    let sim = Self::similarity(y, &self.stats.hits_normed);
                    if sim < self.stats.rand_smpl() {
                        pass_num += 1;
                        true
                    } else {
                        false
                    }
                })
                .collect(),
        };

        self.stats.num_in += n;
        self.stats.num_out += pass_num;
        result
    }

    fn observe<OT: ObserversTuple<I, S>>(&mut self, observers: &OT, input: &I) {
        let observer = observers.match_name::<O>(&self.name).unwrap();
        let map = self.compress(observer.map().unwrap());

        // observe the sample
        self.stats.num_obv += 1;
        self.obv_xs.push(input.bytes().to_vec());
        self.obv_ys.push(map);

        // try train the model
        if self.obv_xs.len() == self.batch_size {
            let xs: Vec<_> = self.obv_xs.iter().map(|x| x.as_slice()).collect();
            let ys: Vec<_> = self.obv_ys.iter().map(|y| y.as_slice()).collect();

            self.stats.update_hits(&ys);

            self.model.train(&xs, &ys, TRN_STEP);
            self.obv_xs.truncate(0);
            self.obv_ys.truncate(0);
        }
    }
}
