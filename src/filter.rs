use libafl::observers::{map::MapObserver, ObserversTuple};
use num_traits::PrimInt;
use std::fmt::Debug;
use std::marker::PhantomData;
use tensorflow as tf;

pub trait AsTenser<T>
where
  T: tf::TensorType,
{
  fn as_tensor(&self) -> Option<tf::Tensor<T>>;
}

impl<S, T> AsTenser<S> for T
where
  T: MapObserver<S>,
  S: tf::TensorType + PrimInt + Copy + Default + Debug,
{
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
  fn run(&mut self, batch: Vec<I>, state: &mut S, coupus_idx: usize) -> (Vec<I>, Vec<f32>);
  /// observe a new sample for the model
  fn observe<OT: ObserversTuple<I, S>>(&mut self, obs: &OT, input: &I);
}

#[allow(unused)]
pub struct CovFilter<I, O, T>
where
  O: MapObserver<T>,
  T: tf::TensorType + PrimInt + Clone + Debug,
{
  /// name used get the MapObserver from ObserversTuple
  name: String,
  batch_size: usize,
  model: tf::SavedModelBundle,
  phantom: PhantomData<(I, O, T)>,
}

impl<I, O, S, T> Filter<I, S> for CovFilter<I, O, T>
where
  O: MapObserver<T>,
  T: tf::TensorType + PrimInt + Clone + Debug,
{
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn run(&mut self, batch: Vec<I>, _state: &mut S, _corpus_id: usize) -> (Vec<I>, Vec<f32>) {
    // TODO: (yun) just for test make all inputs pass the filter
    let mut prob = Vec::with_capacity(batch.len());
    prob.fill(1.0);
    (batch, prob)
  }

  fn observe<OT: ObserversTuple<I, S>>(&mut self, observers: &OT, _input: &I) {
    let observer = observers.match_name::<O>(&self.name).unwrap();
    let _output = observer.as_tensor().unwrap();
    todo!("yun")
  }
}

#[cfg(test)]
mod tests {
  #[test]
  fn test_run_py_script() {
    use std::process::Command;
    // use std::io::{Write, stderr};
    // Command::new("pwd").spawn().unwrap();
    let rst = Command::new("python3")
      .arg("./src/py/create_model.py")
      .args(["-m", "test"])
      .output()
      .expect("Fail run py script");
    let s = String::from_utf8(rst.stdout).unwrap();
    // stderr().write(&rst.stderr).unwrap();
    assert_eq!(s, "test\n".to_string());
  }
}

