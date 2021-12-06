use crate::utils::ChanStore;
use crate::filter::FilterRequest;
use libafl::fuzzer::{Fuzzer, StdFuzzer};
use crossbeam::channel::{ Receiver, Sender};


struct Controller<I, R> {

    flt_req_chan: Sender<FilterRequest>,

}


impl<I, R> Controller<I, R> {

}



