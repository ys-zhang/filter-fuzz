use tensorflow;

pub trait Filter<I,S> {
    /// number of prefered inputs for each run of the filter
    fn batch_size(&self) -> usize;
    /// run a batch, returns the batch itself together with
    /// the probability to do real fuzzing on the target program
    fn run(&mut self, batch: Vec<I>, state: &mut S, coupus_idx: usize) -> (Vec<I>, Vec<f32>);
    /// let the filter model observer real fuzzing results
    fn observe<O>(&mut self, input: &I, output: &O);
}


pub struct CovFilter {
    model: 
}