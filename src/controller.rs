use crate::filter::FilterRequest;
use libafl::fuzzer::{Fuzzer, StdFuzzer};
use crossbeam::channel::{ Receiver, Sender};
use std::marker::PhantomData;

pub struct Controller<I, R> {
    flt_req_chan: Sender<FilterRequest>,
    phantom: PhantomData<(I, R)>,
}


impl<I, R> Controller<I, R> {

    pub fn start(&mut self) {

    }
}



