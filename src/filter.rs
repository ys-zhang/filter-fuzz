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

  num_obv: usize,
  num_in: usize,

  // model: tf::SavedModelBundle,
  phantom: PhantomData<(I, O, T)>,
  
}


impl<I, O, T> CovFilter<I, O, T>
where 
  O: MapObserver<T>,
  T: tf::TensorType + PrimInt + Clone + Debug,
{
  pub fn new(name: &str, batch_size: usize) -> Self {
    Self {
      name: name.to_string(),
      batch_size,
      num_in: 0,
      num_obv: 0,
      phantom: PhantomData,
    }
  }
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
    let prob = batch.iter().map(|_| 1.0).collect();
    self.num_in += batch.len();
    // println!("Filter get inputs: {}", self.num_in);
    (batch, prob)
  }

  fn observe<OT: ObserversTuple<I, S>>(&mut self, observers: &OT, _input: &I) {
    let observer = observers.match_name::<O>(&self.name).unwrap();
    let _output = observer.as_tensor().unwrap();
    self.num_obv += 1;
    if self.num_obv % 1000 == 0 {
      println!("Filter observed {}", self.num_obv);
    }
    // TODO: (yun) observe the sample
  }
}

#[cfg(test)]
mod tests {
  use std::process::Command;
  use std::path;
  use tensorflow as tf;

  const PY: &str = "python3";
  const SCRIPT: &str = "src/py/create_model.py";

  #[test]
  fn test_run_py_script() {
    let rst = Command::new(PY)
      .arg(SCRIPT)
      .args(["-m", "test"])
      .output()
      .expect("Fail run py script");
    let s = String::from_utf8(rst.stdout).unwrap();
    // stderr().write(&rst.stderr).unwrap();
    assert_eq!(s, "test\n".to_string());
  }

  
  fn load_tf_model() -> tf::SavedModelBundle {
    // generate default dense model for 
    let model_name = "test-model";
    let mut model_dir = path::PathBuf::from("./models");
    model_dir.push(model_name);

    if !model_dir.exists() {
      let rst = Command::new(PY)
      .arg(SCRIPT)
      .args(["-m", "dense"])
      .args(["-i", "512"])
      .args(["-o", "2048"])
      .args(["-n", model_name])
      .output()
      .expect("failed to command create model");
    assert!(rst.status.success());
    } 
    

    // try load model
    
    let mut graph = tf::Graph::new();
    let bundle = tf::SavedModelBundle::load(
      &tf::SessionOptions::new(),
      &["serve"],
      &mut graph,
      model_dir,
    ).expect("Failed loading tensorflow keras model");

    bundle
  }

  #[test]
  fn test_load_tf_bundle() {
    let bundle = load_tf_model();
    println!("{:?}", bundle.meta_graph_def().signatures());
  }
}
