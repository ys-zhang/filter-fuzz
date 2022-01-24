use super::{tf, utils, Debug, Filter};
use libafl::{
    bolts::rands::{Rand, StdRand},
    inputs::HasBytesVec,
    observers::{map::MapObserver, ObserversTuple},
    ExecuteInputResult,
};
use num_traits::{cast::cast, PrimInt};
use std::{marker::PhantomData, vec};

// TODO: Hard code parameters
const RND_SEED: u64 = 8944; // random seed for filter
const TRN_EPOCH: usize = 4; // number of epoch for training
const WND_FACTOR: usize = 1 << 4; // filter stat's window size
const PRH_FACTOR: usize = 10; // number of batches for preheat mode train

#[derive(Debug)]
struct CovFilterStats {
    num_in: usize,     // number of inputs fed in
    num_out: usize,    // number of inputs that gets out of the filter
    num_obv: usize,    // number of real execution
    num_ov_in: usize,  // number of oversized input
    num_ov_out: usize, // number of oversized output
    model_in_dim: usize,
    model_out_dim: usize,
}

impl CovFilterStats {
    fn new(model_in_dim: usize, model_out_dim: usize) -> CovFilterStats {
        Self {
            num_in: 0,
            num_out: 0,
            num_obv: 0,
            num_ov_in: 0,
            num_ov_out: 0,
            model_in_dim,
            model_out_dim,
        }
    }

    fn update_in(&mut self, _input: &[u8]) {
        todo!()
    }
}

pub trait Preprocessor<T: PrimInt> {
    fn process_y(&self, map: &[T]) -> Vec<f32>;
    fn update_y(&mut self, map: &[T]);
}

/// EdgeHitInfor stores the current statistics of
/// how edges are hitted during the fuzzing process
#[derive(Debug)]
pub struct Compressor<T: PrimInt> {
    /// indices of edges that have been hit
    hit_indices: Vec<usize>,
    phantom: PhantomData<T>,
}

impl<T: PrimInt> Compressor<T> {
    fn new(model_out_dim: usize) -> Self {
        Self {
            hit_indices: Vec::with_capacity(model_out_dim),
            phantom: PhantomData,
        }
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.hit_indices.len() == self.hit_indices.capacity()
    }
}

/// Compressor keep track of the edges in coverage map that
/// has been hit, it removes edges in coverage map that has not
/// been hit
impl<T: PrimInt> Preprocessor<T> for Compressor<T> {
    /// notice that the dim of the output may be less than
    /// tensorflow model's output_dim.
    ///
    /// Trailing zeros are added when transform vector to tensor
    /// see `Model::new_tensor_from`
    #[inline]
    fn process_y(&self, map: &[T]) -> Vec<f32> {
        // TODO: this might be slow, need to profile
        self.hit_indices
            .iter()
            .map(|&i| cast(map[i]).unwrap())
            .collect()
    }

    /// when `hit_indices` is full, no update will perform
    #[inline]
    fn update_y(&mut self, map: &[T]) {
        // this function is seldom called (only when map is considered
        //   interesting by the fuzzer)
        if self.is_full() {
            return;
        }
        map.iter().enumerate().for_each(|(i, v)| {
            if !v.is_zero() && !self.hit_indices.contains(&i) && !self.is_full() {
                self.hit_indices.push(i);
            }
        });
    }
}

/// Similarity calculate the similarity of coverage maps with
/// a baseline, which summarizes the history of coverage maps,
/// and judge whether a coverage map is considered similar to
/// the baseline
pub trait Similarity {
    fn baseline(&self) -> &[f32];
    fn baseline_mut(&mut self) -> &mut [f32];
    fn update_baseline(&mut self, samples: &[&[f32]]);
    /// calculate the similarity of baseline with sample
    /// `input` and `baseline` must have the same dimension
    fn similarity(&self, input: &[f32]) -> f32;
    /// judge whether an input is similar to the baseline
    fn judge(&mut self, input: &[f32]) -> bool;
}

/// CosSim calculates the Cosine similarity between a sample
/// and the baseline, it keeps a collection of similarity values.
/// It judges whether a sample is considered to be similar to the
/// baseline by sampling from that collection, if the sample's value is
/// greater than the similarity value of the input
pub struct CosSim {
    /// random generator to sample from samples
    rand: StdRand,
    /// normalized sum of observed coverage map
    baseline: Vec<f32>,
    /// sum of observed coverage map
    baseline_unnormed: Vec<f32>,
    /// a collection of observed similarity values
    samples: Vec<f32>,
    next_sample_to_replace: usize,
}

impl CosSim {
    pub fn new(dim: usize, window_size: usize) -> Self {
        Self {
            rand: StdRand::with_seed(RND_SEED),
            baseline: vec![0.0; dim],
            baseline_unnormed: vec![0.0; dim],
            samples: vec![f32::INFINITY; window_size],
            next_sample_to_replace: 0,
        }
    }
}

impl Similarity for CosSim {
    #[inline]
    fn baseline(&self) -> &[f32] {
        &self.baseline
    }

    #[inline]
    fn baseline_mut(&mut self) -> &mut [f32] {
        &mut self.baseline
    }

    /// update cov map hits count, `samples` must have been compressed
    fn update_baseline(&mut self, samples: &[&[f32]]) {
        // add new observed covmap to hits count
        for y in samples {
            for (s, t) in self.baseline_unnormed.iter_mut().zip(y.iter()) {
                *s += *t;
            }
        }
        // update the standardized hits count
        let mean =
            self.baseline_unnormed.iter().sum::<f32>() / (self.baseline_unnormed.len() as f32);
        let centered: Vec<_> = self.baseline_unnormed.iter().map(|&h| h - mean).collect();
        let norm = centered
            .iter()
            .map(|&h| (h - mean) * (h - mean))
            .sum::<f32>()
            .sqrt();
        for (h, ch) in self.baseline.iter_mut().zip(centered.iter()) {
            *h = *ch / norm;
        }
    }

    #[inline]
    fn similarity(&self, input: &[f32]) -> f32 {
        self.baseline().iter().zip(input).map(|(b, a)| a * b).sum()
    }

    /// if the input is considered not similar to the baseline,
    /// then oldest similarity value is replaced with input's similarity value,
    /// this keeps the collection fresh
    #[inline]
    fn judge(&mut self, input: &[f32]) -> bool {
        let sim_val = self.similarity(input);
        if *self.rand.choose(&self.samples) < sim_val {
            true
        } else {
            self.samples[self.next_sample_to_replace] = sim_val;
            self.next_sample_to_replace = (self.next_sample_to_replace + 1) % self.samples.len();
            false
        }
    }
}

pub enum FilterMode {
    Preheat(Samples),
    Ready(Samples),
}

#[derive(Debug)]
pub struct Samples {
    xs: Vec<Vec<u8>>,
    ys: Vec<Vec<f32>>,
}

impl Samples {
    fn new(batch_size: usize) -> Self {
        Self {
            xs: Vec::with_capacity(batch_size),
            ys: Vec::with_capacity(batch_size),
        }
    }

    #[inline]
    fn push(&mut self, x: Vec<u8>, y: Vec<f32>) {
        self.xs.push(x);
        self.ys.push(y);
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.xs.len() == self.xs.capacity()
    }

    fn truncate(&mut self) {
        self.xs.truncate(0);
        self.ys.truncate(0);
    }
}

#[allow(unused)]
pub struct CovFilter<I, J, O, P, T>
where
    O: MapObserver<T>,
    P: Preprocessor<T>,
    J: Similarity,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    /// name used get the MapObserver from ObserversTuple
    name: String,
    batch_size: usize,
    mode: FilterMode,
    stats: CovFilterStats,
    preprocessor: P,
    judge: J,
    model: utils::Model,
    phantom: PhantomData<(I, O, T)>,
}

impl<I, O, T> CovFilter<I, CosSim, O, Compressor<T>, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    pub fn new(name: &str, _map_size: usize, batch_size: usize) -> Self {
        let model = utils::Model::new(name);
        let stats = CovFilterStats::new(model.in_dim, model.out_dim);
        let preprocessor = Compressor::new(model.out_dim);
        let judge = CosSim::new(model.out_dim, WND_FACTOR * batch_size);
        Self {
            name: name.to_string(),
            batch_size,
            mode: FilterMode::Preheat(Samples::new(PRH_FACTOR * batch_size)),
            stats,
            preprocessor,
            judge,
            model,
            phantom: PhantomData,
        }
    }
}

impl<I, J, O, P, T> CovFilter<I, J, O, P, T>
where
    O: MapObserver<T>,
    P: Preprocessor<T>,
    J: Similarity,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    fn is_model_stale(&self) -> bool {
        todo!()
    }

    fn rebuild_model(&mut self) {
        todo!()
    }
}

impl<I, J, O, P, S, T> Filter<I, S> for CovFilter<I, J, O, P, T>
where
    I: HasBytesVec,
    J: Similarity,
    O: MapObserver<T>,
    P: Preprocessor<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn filter(&mut self, batch: &[I], _state: &mut S, _corpus_idx: usize) -> Vec<bool> {
        let n = batch.len();
        let m = self.stats.model_out_dim;

        let mut pass_num: usize = 0;

        let result = match self.mode {
            // in the preheat mode, fuzz all inputs
            FilterMode::Preheat(_) => {
                pass_num = n;
                vec![true; pass_num]
            }

            FilterMode::Ready(_) => {
                let model_in_dim = self.stats.model_in_dim;
                let xs: Vec<&[u8]> = batch.iter().map(|x| x.bytes()).collect();
                let ys_normed = unsafe { self.model.predict_normed(&xs) };
                
                ys_normed
                    .windows(m)
                    .zip(xs)
                    .map(|(y, x)| {
                        // oversized inputs are always fuzzed
                        if x.len() > model_in_dim || !self.judge.judge(y) {
                            pass_num += 1;
                            true
                        } else {
                            false
                        }
                    })
                    .collect()
            }
        };

        self.stats.num_in += n;
        self.stats.num_out += pass_num;
        result
    }

    fn observe<OT: ObserversTuple<I, S>>(
        &mut self,
        observers: &OT,
        input: &I,
        result: ExecuteInputResult,
    ) {
        let observer = observers.match_name::<O>(&self.name).unwrap();
        let full_map = observer.map().unwrap();

        // observe the sample
        self.stats.num_obv += 1;
        match self.mode {
            FilterMode::Preheat(ref mut train_samples) => {
                self.preprocessor.update_y(full_map);
                let map = self.preprocessor.process_y(full_map);

                train_samples.push(input.bytes().to_vec(), map);

                // try train the model
                if train_samples.is_full() {
                    let xs: Vec<_> = train_samples.xs.iter().map(|x| x.as_slice()).collect();
                    let ys: Vec<_> = train_samples.ys.iter().map(|y| y.as_slice()).collect();
                    self.model.train(&xs, &ys, TRN_EPOCH);
                    self.mode = FilterMode::Ready(Samples::new(self.batch_size));
                }
            }
            FilterMode::Ready(ref mut train_samples) => {
                // when the current input is interesting, i.e., a new branch is
                // discovered, try update the edge hit info
                if result != ExecuteInputResult::None {
                    self.preprocessor.update_y(full_map);
                }
                let map = self.preprocessor.process_y(full_map);

                train_samples.push(input.bytes().to_vec(), map);

                // try train the model
                if train_samples.is_full() {
                    let xs: Vec<_> = train_samples.xs.iter().map(|x| x.as_slice()).collect();
                    let ys: Vec<_> = train_samples.ys.iter().map(|y| y.as_slice()).collect();

                    self.model.train(&xs, &ys, TRN_EPOCH);
                    train_samples.truncate();

                    if self.is_model_stale() {
                        self.rebuild_model();
                    }
                }
            }
        }
    }
}
