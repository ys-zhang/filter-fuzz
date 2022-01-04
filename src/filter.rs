use libafl::inputs::HasBytesVec;
use libafl::observers::{map::MapObserver, ObserversTuple};
use num_traits::{cast::cast, PrimInt};
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

    num_obv: usize, // number of real execution
    num_in: usize,  // number of inputs fed in

    obv_xs: Vec<Vec<u8>>,
    obv_ys: Vec<Vec<f32>>,

    model: utils::Model,
    phantom: PhantomData<(I, O, T)>,
}

impl<I, O, T> CovFilter<I, O, T>
where
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    pub fn new(name: &str, batch_size: usize) -> Self {
        let model = utils::Model::new(name);
        let obv_xs = Vec::with_capacity(batch_size);
        let obv_ys = Vec::with_capacity(batch_size);
        Self {
            name: name.to_string(),
            batch_size,

            num_in: 0,
            num_obv: 0,
            obv_xs,
            obv_ys,

            model,
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
        self.num_in += batch.len();
        let xs: Vec<&[u8]> = batch.iter().map(|x| x.bytes()).collect();
        let ys = self.model.predict(&xs);
        let prob = self.filter(ys);
        (batch, prob)
    }

    fn observe<OT: ObserversTuple<I, S>>(&mut self, observers: &OT, input: &I) {
        let observer = observers.match_name::<O>(&self.name).unwrap();
        let map = observer.map().unwrap();
        self.num_obv += 1;
        // observe the sample
        if self.obv_xs.len() == self.batch_size {
            let xs: Vec<_> = self.obv_xs.iter().map(|x| x.as_slice()).collect();
            let ys: Vec<_> = self.obv_ys.iter().map(|y| y.as_slice()).collect();
            self.model.train(&xs, &ys, 32); // TODO: hardcode, train for 32 steps
            self.obv_xs.truncate(0);
            self.obv_ys.truncate(0);
        }
        self.obv_xs.push(input.bytes().to_vec());
        self.obv_ys
            .push(map.iter().map(|&t| cast(t).unwrap()).collect());
    }
}

impl<I, O, T> CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    fn filter(&self, _ys: tf::Tensor<f32>) -> Vec<f32> {
        todo!("yun")
    }
}

mod utils {
    use super::tf;
    use super::tf::Tensor;
    use std::path;
    use std::process::Command;
    use std::ptr;

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

    #[derive(Debug)]
    pub struct Model {
        pub in_dim: usize,
        pub out_dim: usize,
        graph: tf::Graph,
        bundle: tf::SavedModelBundle,
    }

    impl Model {
        pub fn new(name: &str) -> Self {
            let (graph, bundle) = load_tf_model(name);
            let sig = bundle.meta_graph_def().get_signature("train").unwrap();
            let x_shape = sig.get_input("x").unwrap().shape();
            let y_shape = sig.get_input("y").unwrap().shape();
            println!("xshape: {}", x_shape);
            println!("yshape: {}", y_shape);

            Self {
                in_dim: x_shape[1].unwrap() as usize,
                out_dim: y_shape[1].unwrap() as usize,
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

            let mut step = tf::SessionRunArgs::new();
            step.add_feed(&op_x, 0, &xs);
            step.add_target(&op_y_hat);
            let y_hat_tok = step.request_fetch(&op_y_hat, 0);
            session.run(&mut step).unwrap();

            // let mut output_step = tf::SessionRunArgs::new();
            // let y_hat_ix = output_step.request_fetch(&op_y_hat, 0);
            // session.run(&mut output_step).unwrap();

            let y_hat: Tensor<f32> = step.fetch(y_hat_tok).unwrap();
            y_hat
        }

        pub fn train(&self, xs: &[&[u8]], ys: &[&[f32]], nstep: usize) {
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
            for _ in 0..nstep {
                session.run(&mut step).unwrap();
            }
        }

        // create a new tensor and copy the inputs to the tensor
        fn new_tensor_from<T: tf::TensorType>(xs: &[&[T]], dim: usize) -> tf::Tensor<T> {
            let n = xs.len(); // number of samples/inputs
            let mut tensor = tf::Tensor::<T>::new(&[n as u64, dim as u64]);
            let mut ts_ptr = tensor.as_mut_ptr();
            // copy xs to tensor
            unsafe {
                for x in xs {
                    ptr::copy_nonoverlapping(x.as_ptr(), ts_ptr, x.len());
                    ts_ptr = ts_ptr.add(dim);
                }
            }
            tensor
        }
    }
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
