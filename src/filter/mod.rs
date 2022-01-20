use std::fmt::Debug;
use tensorflow as tf;
use num_traits::PrimInt;
use libafl::observers::{map::MapObserver, ObserversTuple};

mod utils;
mod cov;
pub use cov::CovFilter;

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
    /// number of preferred inputs for each run of the filter
    fn batch_size(&self) -> usize;
    /// run a batch, returns the batch itself together with
    /// the probability to do real fuzzing on the target program
    fn run(&mut self, batch: Vec<I>, state: &mut S, corpus_idx: usize) -> (Vec<I>, Vec<f32>);
    /// observe a new sample for the model
    fn observe<OT: ObserversTuple<I, S>>(&mut self, obs: &OT, input: &I);
}


#[cfg(test)]
mod tests {

    use super::tf;
    use super::utils::*;
    use std::process::Command;

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

    #[test]
    fn test_load_tf_bundle() {
        let (_, bundle) = load_tf_model("test-model");
        println!("{:?}", bundle.meta_graph_def().signatures());
    }

    #[test]
    fn test_tf_model() {
        // test model load
        let model = Model::new("test-model");
        assert_eq!(512, model.in_dim);
        assert_eq!(2048, model.out_dim);
        // test predict
        let x: Vec<u8> = vec![0 as u8, 0 as u8, 0 as u8];
        let y_hat = model.predict(&[&x]);
        assert_eq!(
            y_hat.shape(),
            tf::Shape::new(Some(vec![Some(1), Some(model.out_dim as i64)]))
        );

        // test train
    }
}
