use libafl::{
    corpus::CorpusScheduler,
    events::EventManager,
    feedbacks::Feedback,
    fuzzer::StdFuzzer,
    inputs::Input,
    stages::StagesTuple,
    state::{HasClientPerfStats, HasExecutions},
    Error,
};

/// Generator runs in its own thread generate inputs for filter
pub trait Generator<I, R, E, EM, S, ST> {
    /// generate one sample of input.
    fn gen_one_x(&mut self, stages: &mut ST, state: &mut S, manager: &mut EM) -> Result<(I, R), Error>;

    /// generate one sample of input-response pair
    fn gen_one(
        &mut self,
        stages: &mut ST,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(I, R), Error>;

    /// generate `n` sample of input.
    #[allow(unused)] // TODO: check this later
    fn gen_n_x(
        &mut self,
        n: usize,
        stages: &mut ST,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(Vec<I>, Vec<R>), Error> {
        todo!("(zys) need default impl")
    }

    /// generate `n` sample of input-response pair
    #[allow(unused)] // TODO: check this later
    fn gen_n(
        &mut self,
        n: usize,
        stages: &mut ST,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(Vec<I>, Vec<R>), Error> {
        todo!("(zys) need default impl")
    }
}

impl<I, R, E, EM, ST, C, CS, F, OF, OT, S, SC> Generator<I, R, E, EM, S, ST>
    for StdFuzzer<C, CS, F, I, OF, OT, S, SC>
where
    CS: CorpusScheduler<I, S>,
    EM: EventManager<E, I, S, Self>,
    F: Feedback<I, S>,
    I: Input,
    S: HasExecutions + HasClientPerfStats,
    OF: Feedback<I, S>,
    ST: StagesTuple<E, EM, S, Self>,
{
    #[inline]
    #[allow(unused)] // TODO: check this later
    fn gen_one(
        &mut self,
        stages: &mut ST,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(I, R), Error> {
        todo!("zys")
    }

    #[inline]
    #[allow(unused)] // TODO: check this later
    fn gen_one_x(&mut self, stages: &mut ST, state: &mut S, manager: &mut EM) -> Result<(I, R), Error> {
        todo!("zys")
    }
}
