use crate::{
    filter::{Filter, FilterController, FilterRequest},
    utils::ChanStore,
};
use crossbeam::channel::Sender;
use std::thread;
use std::marker::PhantomData;

pub struct Controller<I, R, F> {
    flt_thread: Option<thread::JoinHandle<()>>,
    chan_store: ChanStore<I, R>,
    phantom: PhantomData<F>,
}

impl<I, R, F> Controller<I, R, F>
where
    F: Filter<I, R> + Send + 'static,
    I: Send,
    R: Send,
{
    pub fn new() -> Self {
        todo!("zys")
    }

    pub fn start_filter(&mut self, filter: F) -> Result<(), String> {
        if self.flt_thread.is_some() {
            Err("Filter thread already exists".to_string())
        } else {
            let flt_req_ch = self.chan_store.flt_req_rx.take().unwrap();
            let smpl_ch = self.chan_store.smpl_rx.take().unwrap();
            let flt_in = self.chan_store.flt_in_rx.take().unwrap();
            let flt_out = self.chan_store.flt_out_tx.take().unwrap();
            self.flt_thread = Some(thread::spawn(move || {
                filter.run(
                    flt_req_ch,
                    smpl_ch,
                    flt_in,
                    flt_out,
                )
            }));
            Ok(())
        }
    }
}

// impl<I, R, F> FilterController<I, R> for Controller<I, R, F> {
//     #[inline]
//     fn get_flt_req_tx(&self) -> &Sender<FilterRequest> {
//         &self.chan_store.flt_req_tx.unwrap()
//     }
// }
