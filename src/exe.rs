use libafl::Error as AFLError;
use crossbeam::channel::{Sender, Receiver};

pub trait Executor<I, R> {
    fn exec(&mut self, input: &I) -> Result<R, AFLError>;

    fn run(&mut self, flt_out: Receiver<I>, smpl_ch: Sender<(I, R)>) {
        for input in flt_out {
            if let Ok(rst) = self.exec(&input) {
                smpl_ch.send((input, rst)).expect("Failed try sending samples");
            }
        }
    }
}
