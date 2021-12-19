pub mod stages;
pub mod filter;

use libafl::{
  bolts::current_time,
  corpus::CorpusScheduler,
  events::{EventManager, ProgressReporter},
  feedbacks::Feedback,
  fuzzer::{HasCorpusScheduler, StdFuzzer},
  inputs::Input,
  state::{HasClientPerfMonitor, HasExecutions},
  executors::HasObservers,
  observers::ObserversTuple,
  Error,
};

use std::time::Duration;
use crate::stages::*;
use crate::filter::Filter;


/// Trait mimics libafl::fuzzer::Fuzzer;
pub trait FilterFuzzer<E, EM, F, I, S, ST>
where
  I: Input,
  EM: ProgressReporter<I>,
  S: HasExecutions + HasClientPerfMonitor,
{
  /// same parameter with libafl
  const STATS_TIMEOUT_DEFAULT: Duration = Duration::from_millis(3 * 1000);

  fn filter_fuzz_one(
    &mut self,
    stages: &mut ST,
    executor: &mut E,
    filter: &mut F,
    state: &mut S,
    manager: &mut EM,
  ) -> Result<usize, Error>;

  fn filter_fuzz_loop(
    &mut self,
    stages: &mut ST,
    executor: &mut E,
    filter: &mut F,
    state: &mut S,
    manager: &mut EM,
  ) -> Result<usize, Error> {
    let mut last = current_time();
    let monitor_timeout = Self::STATS_TIMEOUT_DEFAULT;
    loop {
      self.filter_fuzz_one(stages, executor, filter, state, manager)?;
      last = manager.maybe_report_progress(state, last, monitor_timeout)?;
    }
  }
}

impl<C, CS, E, EM, F, FLT, I, OF, OT, S, ST, SC> FilterFuzzer<E, EM, FLT, I, S, ST>
  for StdFuzzer<C, CS, F, I, OF, OT, S, SC>
where
  CS: CorpusScheduler<I, S>,
  E: HasObservers<I, OT, S>,
  EM: EventManager<E, I, S, Self>,
  F: Feedback<I, S>,
  FLT: Filter<I, S>,
  I: Input,
  S: HasClientPerfMonitor + HasExecutions,
  OF: Feedback<I, S>,
  OT: ObserversTuple<I, S>,
  ST: FilterStagesTuple<E, EM, FLT, I, OT, S, Self>,
{
  fn filter_fuzz_one(
    &mut self,
    stages: &mut ST,
    executor: &mut E,
    filter: &mut FLT,
    state: &mut S,
    manager: &mut EM,
  ) -> Result<usize, Error> {
    // Get the next index from the scheduler
    let idx = self.scheduler().next(state)?;
    // Execute all stages
    stages.perform_all(self, executor, filter, state, manager, idx)?;
    // Execute the manager
    manager.process(self, state, executor)?;
    Ok(idx)
  }
}


#[cfg(test)]
mod tests {}
