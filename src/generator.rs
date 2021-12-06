use libafl::{
    Error,
};

/// Generator runs in its own thread generate inputs for filter
pub trait InputGenerator<I, E, EM, S, ST> {
    fn gen_one(&mut self, stages: &mut ST, state: &mut S, manager: &mut EM) -> Result<I, Error>;
    fn gen_n(&mut self, n: usize, stages: &mut ST, state: &mut S, manager: &mut EM) -> Vec<I> {
        (1..n)
            .filter_map(|_| self.gen_one(stages, state, manager).ok())
            .collect()
    }
}

pub trait SampleGenerator<I, R, E, EM, S, ST> {
    /// generate one sample of input-response pair
    fn gen_one(
        &mut self,
        input: I,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(I, R), Error>;

    /// generate `n` sample of input-response pair
    fn gen_n(
        &mut self,
        inputs: Vec<I>,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> (Vec<I>, Vec<R>) {
        let (is, rs): (Vec<_>, Vec<_>) = inputs
            .into_iter()
            .filter_map(|i| self.gen_one(i, executor, state, manager).ok())
            .unzip();
        (is, rs)
    }
}
