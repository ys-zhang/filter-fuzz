use libafl::{fuzzer::Fuzzer, Error as AFLError};
use std::sync::mpsc;

pub enum Error {
   AFL(AFLError),
}

/// SampleController forces the ability to collect samples
pub trait SampleController<I, R> {
    fn take_receiver(&mut self) -> Option<mpsc::Receiver<(I, R)>>;
    fn take_sender(&mut self) -> Option<mpsc::Sender<(I, R)>>;
}

pub trait Generator<E, EM, I, R, S, ST> {
    fn gen_one_sample(
        &mut self,
        stages: &mut ST,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(I, R), Error>;
}

pub trait FilterController<I, R> {
    fn is_ready(&self) -> bool;
}

pub struct StdController<I, R> {
    smpl_sender: Option<mpsc::Sender<(I, R)>>,
    smpl_receiver: Option<mpsc::Receiver<(I, R)>>,
}

impl<I, R> SampleController<I, R> for StdController<I, R> {
    fn take_receiver(&mut self) -> Option<mpsc::Receiver<(I, R)>> {
        self.smpl_receiver.take()
    }
    fn take_sender(&mut self) -> Option<mpsc::Sender<(I, R)>> {
        self.smpl_sender.take()
    }
}

