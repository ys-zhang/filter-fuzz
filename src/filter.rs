use crossbeam::channel::{select, Receiver, SendError, Sender};
use libafl::bolts::rands::{Rand, StdRand};
use std::marker::PhantomData;
use std::path::PathBuf;
// use tensorflow::Session;

// Message send from controller to modules runs in their own threads
// such as Generator, Filter and Fuzzer
/// Filter Control Message
#[derive(Debug)]
pub enum FltCtrlMsg {
    StTrn,                     // start training
    LdTFMdl(PathBuf),          // load tensorflow model
    GtMdlInfo(Sender<String>), // get model info
    GtFltSt(Sender<FltSt>),    // get filter state
}

/// Filter State
#[derive(Debug)]
pub enum FltSt {
    Trn,   // traning
    Flt,   // filtering
}

// FilterController controls the filter model (which runs in the filter thread)
//    in the main thread.
pub trait FilterController<I, R> {
    fn send_request<T>(&mut self, req: FltCtrlMsg) -> Result<T, SendError<FltCtrlMsg>>;
    fn get_state(&self) -> Result<FltSt, SendError<FltCtrlMsg>> {
        todo!("Yun")
    }
    fn load_tf_model(&mut self, path: PathBuf) -> Result<(), SendError<FltCtrlMsg>> {
        self.send_request::<()>(FltCtrlMsg::LdTFMdl(path))
    }
}

pub trait Filter<X, Y> {
    fn flt_one(&mut self, x: X) -> Option<X>;
    fn flt(&mut self, xs: Vec<X>) -> Vec<X> {
        // `into_iter` returns value but `iter` returns ref
        xs.into_iter().filter_map(|x| self.flt_one(x)).collect()
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

    /// run filter loop
    fn run(
        &mut self,
        req_ch: Receiver<FltCtrlMsg>,
        smpl_ch: Receiver<(X, Y)>,
        flt_in: Receiver<X>,
        flt_out: Sender<X>,
    ) {
        loop {
            select! {
              recv(req_ch) -> _req => todo!("Yun"),
              recv(smpl_ch) -> _xy => todo!("Yun"),
              recv(flt_in) -> _x => todo!("Yun"),
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
    fn flt_one(&mut self, x: X) -> Option<X> {
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
