use crossbeam::channel::{ bounded, unbounded, Receiver, Sender};
use crate::filter::FilterRequest;

/// Keeps channels for sending samples and massage passing
#[allow(unused)] // TODO: check this later
pub struct ChanStore<X, Y> {
    // send/recv inputs
    x_tx: Option<Sender<X>>,
    x_rx: Option<Receiver<X>>,
    // send/recv input-response pair
    xy_tx: Option<Sender<(X, Y)>>,
    xy_rx: Option<Receiver<(X, Y)>>,
    // control msg send to filter
    flt_req_tx: Option<Sender<FilterRequest>>,
    flt_req_rx: Option<Receiver<FilterRequest>>,
}

#[allow(unused)] // TODO: check this later
impl<X, Y> ChanStore<X, Y> {
    pub fn new() -> Self {
        let (x_tx, x_rx) = unbounded();
        let (xy_tx, xy_rx) = unbounded();
        let (flt_req_tx, flt_req_rx) = bounded(1);
        Self {
            x_tx: Some(x_tx),
            x_rx: Some(x_rx),
            xy_tx: Some(xy_tx),
            xy_rx: Some(xy_rx),
            flt_req_tx: Some(flt_req_tx),
            flt_req_rx: Some(flt_req_rx),
        }
    }
}
