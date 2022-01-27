use libafl::observers::ObserversTuple;
use libafl::ExecuteInputResult;
use std::fmt::Debug;

pub mod cov;
mod utils;

pub trait Filter<I, S> {
    /// number of preferred inputs for each run of the filter
    fn batch_size(&self) -> usize;

    fn filter(&mut self, batch: &[I], state: &mut S, corpus_idx: usize) -> Vec<bool>;
    /// observe a new sample for the model
    fn observe<OT: ObserversTuple<I, S>>(
        &mut self,
        obs: &OT,
        input: &I,
        result: ExecuteInputResult,
    );
}
