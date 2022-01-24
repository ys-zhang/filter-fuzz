use crate::filter::Filter;
use libafl::{
    corpus::Corpus,
    executors::HasObservers,
    fuzzer::Evaluator,
    inputs::Input,
    mutators::Mutator,
    observers::ObserversTuple,
    stages::{MutationalStage, StdMutationalStage},
    state::{HasClientPerfMonitor, HasCorpus, HasRand},
    Error,
};
use std::iter::Iterator;

#[inline]
fn new_batch<I, M, S>(
    mutator: &mut M,
    state: &mut S,
    stage_idxes: impl Iterator<Item = i32>,
    seed: &I,
) -> Vec<I>
where
    I: Input,
    M: Mutator<I, S>,
{
    stage_idxes
        .map(|i| {
            let mut input = seed.clone();
            mutator
                .mutate(state, &mut input, i)
                .expect("Fail mutate seed");
            input
        })
        .collect()
}

pub trait FilterStage<E, EM, F, OT, S, Z> {
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

pub trait FilterStagesTuple<E, EM, F, OT, S, Z> {
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

impl<E, EM, F, OT, S, Z> FilterStagesTuple<E, EM, F, OT, S, Z> for () {
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

impl<E, EM, F, OT, S, Z, Head, Tail> FilterStagesTuple<E, EM, F, OT, S, Z> for (Head, Tail)
where
    Head: FilterStage<E, EM, F, OT, S, Z>,
    Tail: FilterStagesTuple<E, EM, F, OT, S, Z>,
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

impl<E, EM, F, I, M, OT, S, Z> FilterStage<E, EM, F, OT, S, Z>
    for StdMutationalStage<E, EM, I, M, S, Z>
where
    E: HasObservers<I, OT, S>,
    F: Filter<I, S>,
    I: Input,
    M: Mutator<I, S>,
    OT: ObserversTuple<I, S>,
    S: HasClientPerfMonitor + HasCorpus<I> + HasRand,
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
        // copy from StdMutationalStage::perform_mutational
        let num = self.iterations(state, corpus_idx)?;
        let batch_size = filter.batch_size();
        let num_batch = num / batch_size;

        let seed = state
            .corpus()
            .get(corpus_idx)?
            .borrow_mut()
            .load_input()?
            .clone();
        let mutator = self.mutator_mut();

        // foreach batch of inputs to be feed into filter
        for b in 0..num_batch {
            let stage_idxes = (b * batch_size) as i32..((b + 1) * batch_size) as i32;
            let batch = new_batch(mutator, state, stage_idxes.clone(), &seed);
            let filter_rst = filter.filter(&batch, state, corpus_idx);
            for ((input, i), pass) in batch.iter().zip(stage_idxes).zip(filter_rst) {
                if !pass {
                    continue;
                }
                //TODO: (yun) execution moves the input, try another approach?
                if let Ok((result, corpus_idx)) =
                    fuzzer.evaluate_input(state, executor, manager, input.clone())
                {
                    // feedback samples to filter
                    filter.observe(executor.observers(), input, result);
                    mutator.post_exec(state, i, corpus_idx)?;
                }
            }
        }

        // last one may not be a full batch
        if num % batch_size != 0 {
            let stage_idxes = (num_batch * batch_size) as i32..num as i32;
            let batch = new_batch(mutator, state, stage_idxes.clone(), &seed);

            for ((input, i), pass) in batch
                .iter()
                .zip(stage_idxes)
                .zip(filter.filter(&batch, state, corpus_idx))
            {
                if !pass {
                    continue;
                }
                //TODO: (yun) execution moves the input, try another approach?
                if let Ok((result, corpus_idx)) =
                    fuzzer.evaluate_input(state, executor, manager, input.clone())
                {
                    // feedback samples to filter
                    filter.observe(executor.observers(), input, result);
                    mutator.post_exec(state, i, corpus_idx)?;
                }
            }
        }
        Ok(())
    }
}
