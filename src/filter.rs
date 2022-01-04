use libafl::inputs::HasBytesVec;
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
    /// number of preferred inputs for each run of the filter
    fn batch_size(&self) -> usize;
    /// run a batch, returns the batch itself together with
    /// the probability to do real fuzzing on the target program
    fn run(&mut self, batch: Vec<I>, state: &mut S, corpus_idx: usize) -> (Vec<I>, Vec<f32>);
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

    // model: utils::Model,
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
            // model: utils::Model::new(name: &str, in_dim: usize, out_dim: usize),
            phantom: PhantomData,
        }
    }
}

impl<I, O, S, T> Filter<I, S> for CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn run(&mut self, batch: Vec<I>, _state: &mut S, _corpus_id: usize) -> (Vec<I>, Vec<f32>) {
        // TODO: (yun) just for test make all inputs pass the filter
        self.num_in += batch.len();

        // create the tensor for inputs
        let xs: Vec<&[u8]> = batch.iter().map(|x| x.bytes()).collect();
        // TODO:
        // let xs = tf::Tensor::

        let prob = batch.iter().map(|_| 1.0).collect();
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

mod utils {
    use super::tf;
    use std::mem::size_of;
    use std::path;
    use std::process::Command;
    use std::ptr;
    use tensorflow::{SignatureDef, Tensor};

    pub const PY: &str = "python3";
    pub const SCRIPT: &str = "src/py/create_model.py";
    pub const MODEL_DIR: &str = "models";

    pub fn load_tf_model(model_name: &str) -> (tf::Graph, tf::SavedModelBundle) {
        // generate default dense model for
        let mut model_dir = path::PathBuf::from(MODEL_DIR);
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
        )
        .expect("Failed loading tensorflow keras model");

        (graph, bundle)
    }

    pub struct Model {
        name: String,
        in_dim: usize,
        out_dim: usize,
        graph: tf::Graph,
        bundle: tf::SavedModelBundle,
    }

    impl Model {
        pub fn new(name: &str, in_dim: usize, out_dim: usize) -> Self {
            let (graph, bundle) = load_tf_model(name);
            Self {
                name: name.to_string(),
                in_dim,
                out_dim,
                graph,
                bundle,
            }
        }

        pub fn predict(&self, xs: &[&[u8]]) -> Tensor<f32> {
            let xs = Self::new_tensor_from(xs, self.in_dim);
            let session = &self.bundle.session;
            let sig = self
                .bundle
                .meta_graph_def()
                .get_signature("predict")
                .unwrap();
            let x_info = sig.get_input("x").unwrap();
            let y_hat_info = sig.get_output("y_hat").unwrap();
            let op_x = self
                .graph
                .operation_by_name(&x_info.name().name)
                .unwrap()
                .unwrap();
            let op_y_hat = self
                .graph
                .operation_by_name(&y_hat_info.name().name)
                .unwrap()
                .unwrap();

            let mut run_args = tf::SessionRunArgs::new();
            run_args.add_feed(&op_x, 0, &xs);
            run_args.add_target(&op_y_hat);
            session.run(&mut run_args).unwrap();

            let mut output_step = tf::SessionRunArgs::new();
            let y_hat_ix = output_step.request_fetch(&op_y_hat, 0);
            session.run(&mut output_step).unwrap();

            let y_hat: Tensor<f32> = output_step.fetch(y_hat_ix).unwrap();
            y_hat
        }

        pub fn train(&self, xs: &[&[u8]], ys: &[&[u8]]) {
            const N_STEP: usize = 100; // number of train steps

            let xs = Self::new_tensor_from(xs, self.in_dim);
            let ys = Self::new_tensor_from(ys, self.out_dim);

            let sig = self.bundle.meta_graph_def().get_signature("train").unwrap();
            let x_info = sig.get_input("x").unwrap();
            let y_info = sig.get_input("y").unwrap();
            let loss_info = sig.get_output("loss").unwrap();
            let op_x = self
                .graph
                .operation_by_name_required(&x_info.name().name)
                .unwrap();
            let op_y = self
                .graph
                .operation_by_name_required(&y_info.name().name)
                .unwrap();
            let op_loss = self
                .graph
                .operation_by_name_required(&loss_info.name().name)
                .unwrap();

            let mut step = tf::SessionRunArgs::new();
            step.add_feed(&op_x, 0, &xs);
            step.add_feed(&op_y, 0, &ys);
            step.add_target(&op_loss);
            let session = &self.bundle.session;
            for _ in 0..N_STEP {
                session.run(&mut step).unwrap();
            }
        }
    }

    // private methods
    impl Model {
        // create a new tensor and copy the inputs to the tensor
        fn new_tensor_from<T: tf::TensorType>(xs: &[&[T]], dim: usize) -> tf::Tensor<T> {
            let n = xs.len(); // number of samples/inputs
            let mut tensor = tf::Tensor::<T>::new(&[n as u64, dim as u64]);
            let mut ts_ptr = tensor.as_mut_ptr();
            // copy xs to tensor
            unsafe {
                for x in xs {
                    ptr::copy_nonoverlapping(x.as_ptr(), ts_ptr, size_of::<T>() * x.len());
                    ts_ptr = ts_ptr.add(dim);
                }
            }
            tensor
        }
    }
}

#[cfg(test)]
mod tests {

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
        const IN_DIM: usize = 512;
        const OUT_DIM: usize = 2048;

        let model = Model::new("test-model", IN_DIM, OUT_DIM);
        let x: Vec<u8> = vec![0 as u8, 0 as u8, 0 as u8];
        let y_hat = model.predict(&[&x]);

        println!("y_hat: {}", y_hat);
    }
}
