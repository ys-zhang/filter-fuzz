use tensorflow as tf;
use libafl::observers::{
  map::MapObserver,
  ObserversTuple,
};
use num_traits::PrimInt;
use std::fmt::Debug;
use std::marker::PhantomData;

pub trait AsTensor {
  type TenserType: tf::TensorType;
  fn as_tensor(&self) -> Option<tf::Tensor<Self::TenserType>>;
}

impl<S, T> AsTensor for T 
where T: MapObserver<S>,
      S: tf::TensorType + PrimInt + Copy + Default + Debug,
{
  Self::TensorType = S;
  fn as_tensor(&self) -> Option<tf::Tensor<S>> {
     let map = self.map()?;
     tf::Tensor::new(&[map.len() as u64]).with_values(map).ok()
  }
}

pub trait Filter<I, S> {
    /// number of prefered inputs for each run of the filter
    fn batch_size(&self) -> usize;
    /// run a batch, returns the batch itself together with
    /// the probability to do real fuzzing on the target program
    fn run(
      &mut self, 
      batch: Vec<I>, 
      state: &mut S, 
      coupus_idx: usize
    ) -> (Vec<I>, Vec<f32>);

    fn observe<OT: ObserversTuple<I, S>>(
      &mut self,
      observers: &OT,
      name: &str, 
      input: &I
    );

}

#[allow(unused)]
pub struct CovFilter<I, O> {
    batch_size: usize,
    model: tf::SavedModelBundle,
    phantom: PhantomData<(I, O)>,
}

impl<I, O, S> Filter<I, S> for CovFilter<I, O> 
where O: MapObserver<u8>,
{
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn run(
    &mut self,
    batch: Vec<I>,
    state: &mut S,
    _corpus_id: usize,
  ) -> (Vec<I>, Vec<f32>) {
    // TODO: (yun) now all pass
    let mut prob = Vec::with_capacity(batch.len());
    prob.fill(1.0);
    (batch, prob)
  }

  fn observe<OT: ObserversTuple<I, S>>(
    &mut self,
    observers: &OT,
    name: &str, 
    input: &I
  ) {
    let observer = observers.match_name::<O>(name);
    todo!("yun")
  }
} 
