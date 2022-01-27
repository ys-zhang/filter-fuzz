use super::{utils, Debug, Filter};
use libafl::{
    bolts::rands::{Rand, StdRand},
    inputs::HasBytesVec,
    observers::{map::MapObserver, ObserversTuple},
    ExecuteInputResult,
};
use num_traits::PrimInt;
use scopeguard::defer;
use std::{marker::PhantomData, vec};
use tch::{nn::Module, Device, IndexOp, Kind, Tensor, TensorIndexer};

// TODO: Hard code parameters
const RND_SEED: u64 = 8944; // random seed for filter
const TRN_EPOCH: usize = 4; // number of epoch for training
const WND_FACTOR: usize = 1 << 4; // filter stat's window size
const PRH_FACTOR: usize = 10; // number of batches for preheat mode train

#[derive(Debug)]
struct CovFilterStats {
    batch_size: usize, // number of samples in a train/pred batch
    num_in: usize,     // number of inputs fed in
    num_out: usize,    // number of inputs that gets out of the filter
    num_obv: usize,    // number of real execution
    num_ov_in: usize,  // number of oversized input
    num_ov_out: usize, // number of oversized output
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
    fn proc(&self, map: Tensor) -> Tensor;
    /// tensor `map`'s kind must be 2-D integer type
    unsafe fn update(&mut self, maps: &Tensor);
    /// update a single
    fn update_raw(&mut self, map: &[u8]) {
        let ts = Tensor::of_blob(
            map.as_ptr(),
            &[map.len() as i64],
            &[1],
            Kind::Uint8,
            Device::Cpu,
        )
        .view((1, map.len() as i64));
        unsafe {
            self.update(&ts);
        } 
    }

    fn proc_raw(&self, map: &[u8]) -> Tensor {
        let map_f: Vec<f32> = map.iter().map(|&e| e as f32).collect();
        let ts = Tensor::of_blob(
            map.as_ptr() as *const u8,
            &[map.len() as i64],
            &[1],
            Kind::Float,
            Device::Cpu,
        )
        .view((1, map.len() as i64));
        self.proc(ts)
    }
}

/// a dummy preprocessor
impl Preprocessor for () {
    #[inline]
    fn proc(&self, map: Tensor) -> Tensor {
        map
    }
    unsafe fn update(&mut self, _map: &Tensor) {}
    fn update_raw(&mut self, _map: &[u8]) {}
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
    indices: Vec<i64>,
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

    fn indexer(&self) -> TensorIndexer {
        unsafe {
            TensorIndexer::IndexSelect(Tensor::of_blob(
                self.indices.as_ptr() as *const u8,
                &[self.indices.len() as i64],
                &[1],
                Kind::Int64,
                Device::Cpu,
            ))
        }
    }

    fn pad(&self, ts: Tensor) -> Tensor {
        if self.is_full() {
            ts
        } else {
            let left = self.indices.capacity() - self.indices.len();
            ts.zero_pad1d(0, left as i64)
        }
    }
}

/// EdgeTracker keep track of the edges in coverage map that
/// has been hit, it removes edges in coverage map that has not
/// been hit
impl Preprocessor for EdgeTracker {
    /// notice that the dim of the output may be less than
    /// tensorflow model's output_dim.
    ///
    /// Trailing zeros are added when transform vector to tensor
    #[inline]
    fn proc(&self, map: Tensor) -> Tensor {
        // TODO: will it be faster to first make map a column major matrix?
        let seletor = self.indexer();
        // remove edges that has not been visited yet
        let extracted = match map.dim() {
            1 => map.i(seletor),
            2 => map.i((.., seletor)),
            _ => panic!("can only process 1 or 2 dim tensors"),
        };
        // pad left with zeros
        self.pad(extracted)
    }

    /// when `self.hit_indices` is full, neglects the update
    /// `map` must be an integer tensor
    #[inline]
    unsafe fn update(&mut self, map: &Tensor) {
        // this function is seldom called (only in Preheat mode
        //    or when map is considered interesting by the fuzzer)
        if self.is_full() {
            return;
        }
        let hit: Vec<bool> = map.any_dim(0, false).into();
        (0 as i64..).zip(hit).for_each(|(i, h)| {
            if h && !self.is_full() && !self.indices.contains(&i) {
                self.indices.push(i);
            }
        });
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
            baseline: ts,
            raw_baseline: ts.copy(),
            samples: vec![f32::INFINITY; window_size],
            i_smpl: 0,
        }
    }

    pub fn similarity(&self, ys: &Tensor) -> Tensor {
        utils::ops::standardize(ys).dot(&self.baseline)
    }

    /// randomly choose n similarity samples from history
    pub fn sample_n(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|i| *self.rand.choose(&self.samples)).collect()
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
        self.baseline = utils::ops::standardize(&self.raw_baseline);
        // update sample window
        let sim: Vec<f32> = self.similarity(ys).into();
        sim.iter().for_each(|&s| {
            self.samples[self.i_smpl] = s;
            self.i_smpl = (self.i_smpl + 1) % self.samples.len();
        });
    }
}

type TrainBatch = (Tensor, Tensor);

pub enum FilterMode {
    Preheat(Vec<TrainBatch>),
    Ready(TrainBatch),
}

#[allow(unused)]
pub struct CovFilter<I, O, T>
where
    O: MapObserver<T>,
    T: PrimInt + Default + Copy + Debug,
{
    /// name used get the MapObserver from ObserversTuple
    name: String,
    mode: FilterMode,
    stats: CovFilterStats,
    /// preprocess coverage map before training on nn module
    preprocessor: Box<dyn Preprocessor>,
    /// judge whether a prediction is "insteresting"
    judge: Box<dyn Judge>,
    /// the nn module
    model: Box<dyn Module>,
    phantom: PhantomData<(I, O, T)>,
}

impl<I, O, T> CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: PrimInt + Default + Copy + Debug,
{
    pub fn new(
        name: &str,
        model: Box<dyn Module>,
        in_dim: usize,
        out_dim: usize,
        batch_size: usize,
    ) -> Self {
        let stats = CovFilterStats::new(batch_size, in_dim, out_dim);
        let preprocessor = EdgeTracker::new(out_dim);
        let judge = CosSim::new(out_dim, WND_FACTOR * batch_size);
        Self {
            name: name.to_string(),
            mode: FilterMode::Preheat(Vec::with_capacity(PRH_FACTOR)),
            stats,
            preprocessor: Box::new(preprocessor),
            judge: Box::new(judge),
            model,
            phantom: PhantomData,
        }
    }

    pub fn is_model_stale(&self) -> bool {
        todo!()
    }

    fn rebuild_model(&mut self) {
        todo!()
    }
}

impl<I, O, S, T> Filter<I, S> for CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: PrimInt + Default + Copy + Debug,
{
    #[inline]
    fn batch_size(&self) -> usize {
        self.stats.batch_size
    }

    fn filter(&mut self, batch: &[I], _state: &mut S, _corpus_idx: usize) -> Vec<bool> {
        let mut pass_counter: usize = 0;

        defer! {
            self.stats.num_in += batch.len();
            self.stats.num_out += pass_counter;
        }

        // filter mode do not calc gradiant
        let guard = tch::no_grad_guard();

        match self.mode {
            // in preheat mode, fuzz all inputs
            FilterMode::Preheat(_) => {
                pass_counter = batch.len();
                vec![true; pass_counter]
            }

            FilterMode::Ready(_) => {
                let xs = unsafe {
                    // TODO: oversize inputs are truncated, it seems better
                    //       to just let them pass
                    utils::input_batch_to_tensor(batch, self.stats.md_in_dim)
                };
                // TODO: should ys be normalized?
                let ys = self.model.forward(&xs);
                let rst = self.judge.interesting(&ys);
                rst.iter()
                    .filter(|&&pass| pass)
                    .for_each(|_| pass_counter += 1);
                rst
            }
        }
    }

    fn observe<OT: ObserversTuple<I, S>>(
        &mut self,
        observers: &OT,
        input: &I,
        result: ExecuteInputResult,
    ) {
        // oversize input
        if input.bytes().len() > self.stats.md_in_dim {
            self.stats.num_ov_in += 1;
            return;
        }
        let observer = observers.match_name::<O>(&self.name).unwrap();
        let full_map = observer.map().unwrap();
        if result != ExecuteInputResult::None {
            self.preprocessor.update_raw(full_map);
        }
        // compress coverage map by removing unobserved edges
        let map = self.preprocessor.proc(full_map);

        // observe the sample
        self.stats.num_obv += 1;
        match self.mode {
            FilterMode::Preheat(ref mut train_samples) => {
                todo!("train in mode preheat ")
                // self.preprocessor.update_y(full_map);
                // train_samples.push(input.bytes().to_vec(), map);
                // // try train the model
                // if train_samples.is_full() {
                //     let xs: Vec<_> = train_samples.xs.iter().map(|x| x.as_slice()).collect();
                //     let ys: Vec<_> = train_samples.ys.iter().map(|y| y.as_slice()).collect();
                //     unsafe {
                //         self.model.train(&xs, &ys, TRN_EPOCH);
                //     }
                //     self.mode = FilterMode::Ready(Samples::new(self.batch_size));
                // }
            }
            FilterMode::Ready(ref mut train_samples) => {
                todo!("train in mode ready")
                // // when the current input is interesting, i.e., a new branch is
                // // discovered, try update the edge hit info
                // if result != ExecuteInputResult::None {
                //     self.preprocessor.update_y(full_map);
                // }
                // train_samples.push(input.bytes().to_vec(), map);
                // // try train the model
                // if train_samples.is_full() {
                //     let xs: Vec<_> = train_samples.xs.iter().map(|x| x.as_slice()).collect();
                //     let ys: Vec<_> = train_samples.ys.iter().map(|y| y.as_slice()).collect();
                //     unsafe {
                //         self.model.train(&xs, &ys, TRN_EPOCH);
                //     }
                //     train_samples.truncate();
                //
                //     if self.is_model_stale() {
                //         self.rebuild_model();
                //     }
                // }
            }
        }
    }
}
