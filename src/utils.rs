use crossbeam::channel::{ bounded, unbounded, Receiver, Sender};
use crate::filter::FilterRequest;

/// Keeps channels for sending samples and massage passing
#[allow(unused)] // TODO: check this later
pub struct ChanStore<X, Y> {
    // send inputs to filter
    pub flt_in_tx: Option<Sender<X>>,
    pub flt_in_rx: Option<Receiver<X>>,
    // send inputs to fuzzer
    pub flt_out_tx: Option<Sender<X>>,
    pub flt_out_rx: Option<Receiver<X>>,
    // send samples to filter
    pub smpl_tx: Option<Sender<(X, Y)>>,
    pub smpl_rx: Option<Receiver<(X, Y)>>,
    // control msg send to filter
    pub flt_req_tx: Option<Sender<FilterRequest>>,
    pub flt_req_rx: Option<Receiver<FilterRequest>>,
}

#[allow(unused)] // TODO: check this later
impl<X, Y> ChanStore<X, Y> {
    pub fn new() -> Self {
        // TODO: (zys) change to bounded channel?
        let (flt_in_tx, flt_in_rx) = unbounded();
        let (flt_out_tx, flt_out_rx) = unbounded();
        let (smpl_tx, smpl_rx) = unbounded();
        let (flt_req_tx, flt_req_rx) = bounded(1);
        Self {
            flt_in_tx: Some(flt_in_tx),
            flt_in_rx: Some(flt_in_rx),
            flt_out_tx: Some(flt_out_tx),
            flt_out_rx: Some(flt_out_rx),
            smpl_tx: Some(smpl_tx),
            smpl_rx: Some(smpl_rx),
            flt_req_tx: Some(flt_req_tx),
            flt_req_rx: Some(flt_req_rx),
        }
    }
}
