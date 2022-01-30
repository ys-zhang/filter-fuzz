use super::{
    utils::{
        data::{input_batch_to_tensor, SampleBuffer},
        model::Model,
        ops,
    },
    Debug, Filter,
};
use libafl::{
    bolts::rands::{Rand, StdRand},
    inputs::HasBytesVec,
    observers::{map::MapObserver, ObserversTuple},
    ExecuteInputResult,
};
use std::{marker::PhantomData, vec};
use tch::{nn, Device, Kind, Tensor};

// TODO: Hard code parameters
const RND_SEED: u64 = 8944; // random seed for filter
const TRN_EPOCH: usize = 4; // number of epoch for training
const WND_FACTOR: usize = 1 << 4; // filter stat's window size
const PRH_BATCH_NUM: usize = 10; // number of batches for preheat mode train

#[derive(Debug)]
struct CovFilterStats {
    batch_size: usize, // number of samples in a train/pred batch
    num_in: usize,     // number of inputs fed in
    num_out: usize,    // number of inputs that gets out of the filter
    num_obv: usize,    // number of real execution
    num_ov_in: usize,  // number of oversize input
    num_ov_out: usize, // number of oversize output
    md_in_dim: usize,  // input dim of nn module
    md_out_dim: usize, // output dim of nn module
}

impl CovFilterStats {
    fn new(batch_size: usize, in_dim: usize, out_dim: usize) -> CovFilterStats {
        Self {
            batch_size,
            num_in: 0,
            num_out: 0,
            num_obv: 0,
            num_ov_in: 0,
            num_ov_out: 0,
            md_in_dim: in_dim,
            md_out_dim: out_dim,
        }
    }
}

pub trait Preprocessor {
    /// notice that all implementation shall copy the data
    fn proc(&self, map: &[u8]) -> Vec<u8>;

    /// update use a single coverage map
    fn update(&mut self, map: &[u8]) -> bool;
}

/// a dummy preprocessor
impl Preprocessor for () {
    #[inline]
    fn proc(&self, map: &[u8]) -> Vec<u8> {
        map.into()
    }
    fn update(&mut self, _map: &[u8]) -> bool {
        true
    }
}

/// tracking statistics of how edges are hit and
/// compress the coverage map by removing edges
/// based on the info
///
/// TODO: is it better to just integrate in the nn module?
#[derive(Debug)]
pub struct EdgeTracker {
    // indices of edges that have been hit
    // i64 is equiv to torch CPULongType
    // for indexing tensors
    indices: Vec<usize>,
}

impl EdgeTracker {
    pub fn new(out_dim: usize) -> Self {
        let indices = Vec::with_capacity(out_dim);
        Self { indices }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.indices.len() == self.indices.capacity()
    }
}

/// EdgeTracker keep track of the edges in coverage map that
/// has been hit, it removes edges in coverage map that has not
/// been hit
impl Preprocessor for EdgeTracker {
    /// notice that the dim of the output may be less than
    /// tensorflow model's output_dim.
    /// this can be super slow if map do not use a vector as backend
    #[inline]
    fn proc(&self, map: &[u8]) -> Vec<u8> {
        self.indices.iter().map(|&i| map[i]).collect()
    }

    /// when `self.hit_indices` is full, neglects the update
    /// `map` must be an integer tensor
    #[inline]
    fn update(&mut self, map: &[u8]) -> bool {
        // this function is seldom called,
        // only when map is considered interesting by the fuzzer
        if self.is_full() {
            false
        } else {
            map.iter().enumerate().for_each(|(i, &h)| {
                if h > 0 && !self.is_full() && !self.indices.contains(&i) {
                    self.indices.push(i);
                }
            });
            true
        }
    }
}

/// judge whether a coverage map is considered interesting
pub trait Judge {
    /// judge whether an input is interesting
    fn interesting(&mut self, ys: &Tensor) -> Vec<bool>;
    /// update the judge
    fn update(&mut self, ys: &Tensor);
}

/// CosSim calculates the Cosine similarity between a sample
/// and the baseline, it keeps a collection of similarity values.
/// It judges whether a sample is considered to be similar to the
/// baseline by sampling from that collection, if the sample's value is
/// greater than the similarity value of the input
pub struct CosSim {
    /// random generator to sample from samples
    rand: StdRand,
    /// standardized sum of observed coverage map
    baseline: Tensor,
    /// sum of observed coverage map
    raw_baseline: Tensor,
    /// a collection of observed similarity values
    samples: Vec<f32>,
    /// next_sample_to_replace
    i_smpl: usize,
}

impl CosSim {
    pub fn new(dim: usize, window_size: usize) -> Self {
        let ts = Tensor::zeros(&[dim as i64], (Kind::Float, Device::Cpu));
        Self {
            rand: StdRand::with_seed(RND_SEED),
            raw_baseline: ts.copy(),
            baseline: ts,
            samples: vec![f32::INFINITY; window_size],
            i_smpl: 0,
        }
    }

    pub fn similarity(&self, ys: &Tensor) -> Tensor {
        unsafe { ops::standardize(ys).dot(&self.baseline) }
    }

    /// randomly choose n similarity samples from history
    pub fn sample_n(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| *self.rand.choose(&self.samples)).collect()
    }
}

impl Judge for CosSim {
    /// judge whether an input is interesting
    fn interesting(&mut self, ys: &Tensor) -> Vec<bool> {
        let sim_ts = self.similarity(ys); // this is a 1-dim vector
        let sim_vc: Vec<f32> = sim_ts.into();
        sim_vc
            .iter()
            .map(|&s| s < *self.rand.choose(&self.samples))
            .collect()
    }

    /// update the judge
    fn update(&mut self, ys: &Tensor) {
        self.raw_baseline += ys.sum_dim_intlist(&[0], false, Kind::Float);
        self.baseline = unsafe { ops::standardize(&self.raw_baseline) };
        // update sample window
        let sim: Vec<f32> = self.similarity(ys).into();
        sim.iter().for_each(|&s| {
            self.samples[self.i_smpl] = s;
            self.i_smpl = (self.i_smpl + 1) % self.samples.len();
        });
    }
}

pub enum FilterMode {
    Preheat(SampleBuffer),
    Ready(SampleBuffer),
}

#[allow(unused)]
pub struct CovFilter<I, O>
where
    O: MapObserver<u8>,
{
    /// name used get the MapObserver from ObserversTuple
    name: String,
    mode: FilterMode,
    stats: CovFilterStats,
    /// preprocess coverage map before training on nn module
    preprocessor: Box<dyn Preprocessor>,
    /// judge whether a prediction is "interesting"
    judge: Box<dyn Judge>,
    /// the nn module
    model: Model,
    phantom: PhantomData<(I, O)>,
}

impl<I, O> CovFilter<I, O>
where
    I: HasBytesVec,
    O: MapObserver<u8>,
{
    pub fn new(name: &str, model: Model, batch_size: usize) -> Self {
        let stats = CovFilterStats::new(batch_size, model.in_dim(), model.out_dim());
        let preprocessor = EdgeTracker::new(model.out_dim());
        let judge = CosSim::new(model.out_dim(), WND_FACTOR * batch_size);
        let init_mode = FilterMode::Preheat(SampleBuffer::new(
            PRH_BATCH_NUM * batch_size,
            model.in_dim(),
            model.out_dim(),
        ));

        Self {
            name: name.to_string(),
            mode: init_mode,
            stats,
            preprocessor: Box::new(preprocessor),
            judge: Box::new(judge),
            model,
            phantom: PhantomData,
        }
    }

    pub fn is_model_stale(&self) -> bool {
        // TODO: now it always returns false
        false
    }

    fn rebuild_model(&mut self) {
        todo!()
    }
}

impl<I, O, S> Filter<I, S> for CovFilter<I, O>
where
    I: HasBytesVec,
    O: MapObserver<u8>,
{
    #[inline]
    fn batch_size(&self) -> usize {
        self.stats.batch_size
    }

    fn filter(&mut self, batch: &[I], _state: &mut S, _corpus_idx: usize) -> Vec<bool> {
        let mut pass_counter: usize = 0;

        let rst = match self.mode {
            // in preheat mode, fuzz all inputs
            FilterMode::Preheat(_) => {
                pass_counter = batch.len();
                vec![true; pass_counter]
            }

            FilterMode::Ready(_) => {
                let _guard = tch::no_grad_guard(); // no backtracking in this scope
                let xs /* u8-cpu */ = unsafe {
                    // TODO: oversize inputs are truncated, it seems better
                    //       to just let them pass
                    input_batch_to_tensor(batch, self.stats.md_in_dim)
                };
                // TODO: should ys be normalized?
                let ys /* f32-cpu */ = self.model.forward(&xs);
                let rst = self.judge.interesting(&ys);
                rst.iter()
                    .filter(|&&pass| pass)
                    .for_each(|_| pass_counter += 1);
                rst
            }
        };

        // update stats
        self.stats.num_in += batch.len();
        self.stats.num_out += pass_counter;

        rst
    }

    fn observe<OT: ObserversTuple<I, S>>(
        &mut self,
        observers: &OT,
        input: &I,
        result: ExecuteInputResult,
    ) {
        // observe the sample
        self.stats.num_obv += 1;
        // oversize input
        if input.bytes().len() > self.stats.md_in_dim {
            self.stats.num_ov_in += 1;
            return;
        }
        let observer = observers.match_name::<O>(&self.name).unwrap();
        let map = {
            let full_buf = observer.map().unwrap();
            if result != ExecuteInputResult::None {
                // fuzzer find a new coverage, update the preprocessor
                // for new hit edges
                self.preprocessor.update(full_buf);
            }
            // Preprocessor.proc copies the data
            self.preprocessor.proc(full_buf)
        };

        let buffer = match &mut self.mode {
            FilterMode::Preheat(ref mut buffer) => buffer,
            FilterMode::Ready(ref mut buffer) => buffer,
        };

        buffer.push(input.bytes(), &map);
        if buffer.is_full() {
            let data = unsafe { buffer.iter2(self.stats.batch_size) };
            self.model.train(data, TRN_EPOCH);

            match &mut self.mode {
                FilterMode::Preheat(_) => {
                    self.mode = FilterMode::Ready(SampleBuffer::new(
                        self.stats.batch_size,
                        self.stats.md_in_dim,
                        self.stats.md_out_dim,
                    ))
                }
                FilterMode::Ready(ref mut buffer) => buffer.truncate(),
            }
        }

        // match &mut self.mode {
        //     FilterMode::Preheat(ref mut buffer) => {
        //         buffer.push(input.bytes(), &map);
        //         if buffer.is_full() {
        //             let data = unsafe { buffer.iter2(self.stats.batch_size) };
        //             utils::train(&mut self.model, data, TRN_EPOCH);
        //             buffer.truncate();
        //         }
        //     }
        //     // nn module is already trained
        //     FilterMode::Ready(ref mut buffer) => {
        //         buffer.push(input.bytes(), &map);
        //         if buffer.is_full() {
        //             let data = unsafe { buffer.iter2(self.stats.batch_size) };
        //             utils::train(&mut self.model, data, TRN_EPOCH);
        //             buffer.truncate();
        //         }
        //     }
        // }
    }
}
