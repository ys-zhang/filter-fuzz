use crossbeam::channel::{select, Receiver, SendError, Sender};
use libafl::bolts::rands::{Rand, StdRand};
use std::marker::PhantomData;
use std::path::PathBuf;
// use tensorflow::Session;

// Message send from controller to modules runs in their own threads
// such as Generator, Filter and Fuzzer
#[derive(Debug)]
#[allow(unused)] // TODO: check this later
pub enum FilterRequest {
    LoadTFModel(PathBuf),
    StartTraining,
}

#[derive(Debug)]
#[allow(unused)] // TODO: check this later
pub enum FilterState {
    Training,
    Ready,
}

// FilterController controls the filter model (which runs in the filter thread)
//    in the main thread.
pub trait FilterController<I, R> {
    fn get_flt_req_tx(&self) -> &Sender<FilterRequest>;
    fn get_state(&self) -> FilterState;

    #[allow(unused)] // TODO: check this later
    fn load_tf_model(&self, path: PathBuf) -> Result<(), SendError<FilterRequest>> {
        self.get_flt_req_tx().send(FilterRequest::LoadTFModel(path))
    }
}

pub trait Filter<X, Y> {
    fn filter_one(&mut self, x: X) -> Option<X>;
    fn filter(&mut self, xs: Vec<X>) -> Vec<X> {
        // `into_iter` returns value but `iter` returns ref
        xs.into_iter().filter_map(|x| self.filter_one(x)).collect()
    }
    fn obv_one(&mut self, x: &X, y: &Y);
    fn obv(&mut self, xs: &[X], ys: &[Y]) {
        let pairs = xs.iter().zip(ys.iter());
        for (x, y) in pairs {
            self.obv_one(x, y);
        }
    }

    // fn get_curr_model(&self) -> &Session;
    fn load_model(&mut self, path: PathBuf);

    #[allow(unused)]
    /// run filter loop
    fn run(
        &mut self,
        req_ch: Receiver<FilterRequest>,
        smpl_ch: Receiver<(X, Y)>,
        flt_in: Receiver<X>,
        flt_out: Sender<X>,
    ) {
        loop {
            select! {
              recv(req_ch) -> _req => todo!("zys"),
              recv(smpl_ch) -> _xy => todo!("zys"),
              recv(flt_in) -> _x => todo!("zys"),
            }
        }
    }
}

#[derive(Debug)]
pub struct TravialFilter<X, Y> {
    pass_threadshold: u64,
    seed: StdRand,
    smpl_count: usize,

    phantom: PhantomData<(X, Y)>,
}

#[allow(unused)] // TODO: check this later
impl<X, Y> TravialFilter<X, Y> {
    pub fn new(pass_rate: f64) -> Self {
        Self {
            pass_threadshold: ((u64::MAX as f64) * pass_rate) as u64,
            seed: StdRand::with_seed(8944),
            smpl_count: 0,
            phantom: PhantomData,
        }
    }
}

impl<X, Y> Filter<X, Y> for TravialFilter<X, Y> {
    #[inline]
    fn filter_one(&mut self, x: X) -> Option<X> {
        if self.seed.next() <= self.pass_threadshold {
            Some(x)
        } else {
            None
        }
    }

    #[inline]
    fn obv_one(&mut self, _x: &X, _y: &Y) {
        self.smpl_count += 1;
    }

    fn load_model(&mut self, path: PathBuf) {
        println!("TravialFilter load {}", path.to_str().unwrap());
    }
}
