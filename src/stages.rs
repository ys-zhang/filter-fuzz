use crate::filter::Filter;
use libafl::{
    bolts::rands::Rand,
    corpus::Corpus,
    fuzzer::Evaluator,
    inputs::Input,
    mutators::Mutator,
    stages::{MutationalStage, StdMutationalStage},
    state::{HasClientPerfMonitor, HasCorpus, HasRand},
    Error,
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
    F: Filter<I, S>,
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
        // copy from StdMutationalStage::perform_mutational
        let num = self.iterations(state, corpus_idx)?;
        let batch_size = filter.batch_size();
        let num_batch = num / batch_size;

        // foreach batch of inputs to be feed into filter
        for b in 0..num_batch {
            let mut batch = Vec::with_capacity(batch_size);
            for i in b * batch_size..(b + 1) * batch_size {
                let mut input = state
                    .corpus()
                    .get(corpus_idx)?
                    .borrow_mut()
                    .load_input()?
                    .clone();
                self.mutator_mut().mutate(state, &mut input, i as i32)?;
                batch.push(input);
            }
            // runs ML model on the batch
            // returns probability the inputs can pass the filter
            let (batch, pass_prob) = filter.run(batch, state, corpus_idx);
            // fuzz the batch
            for (i, (input, prob)) in
                (b * batch_size..).zip(batch.into_iter().zip(pass_prob.into_iter()))
            {
                if state.rand_mut().next() as f32 <= (u64::MAX as f32) * prob {
                    let (_, corpus_idx) = fuzzer.evaluate_input(state, executor, manager, input)?;
                    self.mutator_mut().post_exec(state, i as i32, corpus_idx)?;
                }
            }
        }

        // last one may not be a full batch
        if num % num_batch != 0 {
            let mut batch = Vec::with_capacity(num % num_batch);
            for i in (num_batch - 1) * batch_size..num {
                let mut input = state
                    .corpus()
                    .get(corpus_idx)?
                    .borrow_mut()
                    .load_input()?
                    .clone();
                self.mutator_mut().mutate(state, &mut input, i as i32)?;
                batch.push(input);
            }
            // runs ML model on the batch
            // returns probability the inputs can pass the filter
            let (batch, pass_prob) = filter.run(batch, state, corpus_idx);
            // fuzz the batch
            for (i, (input, prob)) in
                (num - batch.len()..).zip(batch.into_iter().zip(pass_prob.into_iter()))
            {
                if state.rand_mut().next() as f32 <= (u64::MAX as f32) * prob {
                    let (_, corpus_idx) = fuzzer.evaluate_input(state, executor, manager, input)?;
                    
                    self.mutator_mut().post_exec(state, i as i32, corpus_idx)?;
                }
            }
        }
        Ok(())
    }
}
