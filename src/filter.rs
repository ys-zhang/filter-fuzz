use tensorflow as tf;

pub trait AsTensor<T: tf::TensorType> {
  fn as_tensor(&self) -> tf::Tensor<T>;
}


pub trait HasFilterObservation {
  fn get_observation<O: AsTensor<u8>>(&mut self, state &mut S );
}

pub trait Filter<I,S> {
    /// number of prefered inputs for each run of the filter
    fn batch_size(&self) -> usize;
    /// run a batch, returns the batch itself together with
    /// the probability to do real fuzzing on the target program
    fn run(&mut self, batch: Vec<I>, state: &mut S, coupus_idx: usize) -> (Vec<I>, Vec<f32>);
    /// let the filter model observer real fuzzing results
    fn observe<O: AsTensor<u8>>(&mut self, input: &I, output: &O);
}

#[allow(unused)]
pub struct CovFilter {
    model: tf::SavedModelBundle,
}
