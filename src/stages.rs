use libafl::{
  bolts::rands::Rand,
  corpus::{Corpus},
  inputs::Input,
  fuzzer::Evaluator,
  stages::{MutationalStage, StdMutationalStage},
  state::{HasRand, HasCorpus, HasClientPerfMonitor},
  mutators::Mutator,
  Error
};

pub trait FilterStage<E, F, EM, S, Z> {
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        filter: &mut F,
        state: &mut S,
        manager: &mut EM,
        corpus_idx: usize,
    ) -> Result<(), Error>;
}

pub trait FilterStagesTuple<E, F, EM, S, Z> {
    fn perform_all(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        filter: &mut F,
        state: &mut S,
        manager: &mut EM,
        corpus_idx: usize,
    ) -> Result<(), Error>;
}

impl<E, F, EM, S, Z> FilterStagesTuple<E, F, EM, S, Z> for () {
    fn perform_all(
        &mut self,
        _fuzzer: &mut Z,
        _executor: &mut E,
        _filter: &mut F,
        _state: &mut S,
        _manager: &mut EM,
        _corpus_idx: usize,
    ) -> Result<(), Error> {
        Ok(())
    }
}

impl<E, F, EM, S, Z, Head, Tail> FilterStagesTuple<E, F, EM, S, Z> for (Head, Tail)
where
    Head: FilterStage<E, F, EM, S, Z>,
    Tail: FilterStagesTuple<E, F, EM, S, Z>,
{
    fn perform_all(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        filter: &mut F,
        state: &mut S,
        manager: &mut EM,
        corpus_idx: usize,
    ) -> Result<(), Error> {
        self.0
            .perform(fuzzer, executor, filter, state, manager, corpus_idx)?;
        self.1
            .perform_all(fuzzer, executor, filter, state, manager, corpus_idx)
    }
}

impl<C, E, F, EM, I, M, R, S, Z> FilterStage<E, F, EM, S, Z>
    for StdMutationalStage<C, E, EM, I, M, R, S, Z>
where
    C: Corpus<I>,
    M: Mutator<I, S>,
    I: Input,
    R: Rand,
    S: HasClientPerfMonitor + HasCorpus<C, I> + HasRand<R>,
    Z: Evaluator<E, EM, I, S>,
{
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        filter: &mut F,
        state: &mut S,
        manager: &mut EM,
        corpus_idx: usize,
    ) -> Result<(), Error> {
       
    }
}
