use crate::{
    filter::{Filter, FilterController, FltCtrlMsg, FltSt},
    utils::ChanStore,
};
use std::marker::PhantomData;
use std::thread;
use std::path::PathBuf;
use crossbeam::channel::{Sender, SendError};

pub struct Controller<I, R, F, E> {
    flt_thread: Option<thread::JoinHandle<()>>,
    exe_thread: Option<thread::JoinHandle<()>>,
    chan_store: ChanStore<I, R>,
    phantom: PhantomData<(F, E)>,
}

impl<I, R, F, E> Controller<I, R, F, E>
where
    // `T: 'static` means all refs in type `T` should have static lifetime
    // see https://doc.rust-lang.org/rust-by-example/scope/lifetime/lifetime_bounds.html
    F: Filter<I, R> + Send + 'static,  // Filter
    E: Send + 'static,                 // Executor
    I: Send + 'static,                 // Input
    R: Send + 'static,                 // Observed bitmap as Response
{
    pub fn new() -> Self {
        todo!("Yun")
    }

    /// start the filter thread, the filter object gets moved into the thread.
    pub fn start_flt_thread(&mut self, filter: F) -> Result<(), String> {
        if self.flt_thread.is_some() {
            Err("Filter thread already exists".to_string())
        } else {
            let flt_req_ch = self.chan_store.flt_req_rx.take().unwrap();
            let smpl_ch = self.chan_store.smpl_rx.take().unwrap();
            let flt_in = self.chan_store.flt_in_rx.take().unwrap();
            let flt_out = self.chan_store.flt_out_tx.take().unwrap();
            self.flt_thread = Some(thread::spawn(move || {
                let mut filter = filter;  // moves filter in
                filter.run(flt_req_ch, smpl_ch, flt_in, flt_out);
            }));
            Ok(())
        }
    }

    /// start the thread for fuzzing inputs
    pub fn start_exe_thread(&mut self, _exe: E) -> Result<(), String> {
        todo!("Yun")
    }
}


impl<I, R, F, E> FilterController<I, R> for Controller<I, R, F, E> {
    fn send_request<T>(&mut self, _req: FltCtrlMsg) -> Result<T, SendError<FltCtrlMsg>> {
        todo!("Yun")
    }
}