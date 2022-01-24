use super::tf::{self, Tensor};
use libafl::observers::map::MapObserver;
use num_traits::PrimInt;
use std::{fmt::Debug, path, process::Command, ptr};

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

    /// create a new tensor and copy the inputs to the tensor
    /// the result is a tensor of shape `[xs.len(), dim]`
    /// Elements of `xs` should be less or equal to `dim`, if it is less
    /// than dim, trailing zeros are added in the result tensor 
    /// 
    /// oversized elems are replaced with zero vectors 
    pub unsafe fn new_tensor_from<T: tf::TensorType>(xs: &[&[T]], dim: usize) -> tf::Tensor<T> {
        let n = xs.len(); // number of samples/inputs
        // tensor elems are initialized to zero
        let mut tensor = tf::Tensor::<T>::new(&[n as u64, dim as u64]);
        let mut ts_ptr = tensor.as_mut_ptr();
        // copy xs to tensor
        for x in xs {
            // oversized slices are replaced with zero vectors
            if x.len() <= dim {
                ptr::copy_nonoverlapping(x.as_ptr(), ts_ptr, x.len());
            }
            ts_ptr = ts_ptr.add(dim);
        }
        
        tensor
    }

    /// Get the session of the tensorflow computation graph
    pub fn session(&self) -> &tf::Session {
        &self.bundle.session
    }

    pub fn get_signature(&self, name: &str) -> &tf::SignatureDef {
        self.bundle.meta_graph_def().get_signature(name).unwrap()
    }

    /// elems of `xs` should have len <= model.in_dim
    #[allow(unused)]
    #[inline]
    pub unsafe fn predict(&self, xs: &[&[u8]]) -> (Tensor<f32>, Tensor<f32>) {
        let xs = Self::new_tensor_from(xs, self.in_dim);
        let sig = self.get_signature("predict");

        let x = {
            let info = sig.get_input("x").unwrap();
            self.graph
                .operation_by_name(&info.name().name)
                .unwrap()
                .unwrap()
        };

        let y_hat = {
            let info = sig.get_output("y_hat").unwrap();
            self.graph
                .operation_by_name(&info.name().name)
                .unwrap()
                .unwrap()
        };

        let y_hat_normed = {
            let info = sig.get_output("y_hat_normed").unwrap();
            self.graph
                .operation_by_name(&info.name().name)
                .unwrap()
                .unwrap()
        };

        let mut step = tf::SessionRunArgs::new();
        step.add_feed(&x, 0, &xs);
        step.add_target(&y_hat);
        step.add_target(&y_hat_normed);
        let y_hat_tok = step.request_fetch(&y_hat, 0);
        let y_hat_normed_tok = step.request_fetch(&y_hat_normed, 0);
        self.session().run(&mut step).unwrap();

        let y_hat = step.fetch(y_hat_tok).unwrap();
        let y_hat_normed = step.fetch(y_hat_normed_tok).unwrap();
        (y_hat, y_hat_normed)
    }

    /// elems of `xs` should have len <= model.in_dim
    #[allow(unused)]
    #[inline]
    pub unsafe fn predict_unnormed(&self, xs: &[&[u8]]) -> Tensor<f32> {
        self.predict_and_fetch(xs, "y_hat")
    }

    /// elems of `xs` should have len <= model.in_dim
    #[allow(unused)]
    #[inline]
    pub unsafe fn predict_normed(&self, xs: &[&[u8]]) -> Tensor<f32> {
        self.predict_and_fetch(xs, "y_hat_normed")
    }

    /// elems of `xs` should have len <= model.in_dim
    pub unsafe fn predict_and_fetch(&self, xs: &[&[u8]], output_name: &str) -> Tensor<f32> {
        let xs = Self::new_tensor_from(xs, self.in_dim);
        let sig = self.get_signature("predict");
        let x = {
            let info = sig.get_input("x").unwrap();
            self.graph
                .operation_by_name(&info.name().name)
                .unwrap()
                .unwrap()
        };
        let y = {
            let info = sig.get_output(output_name).unwrap();
            self.graph
                .operation_by_name(&info.name().name)
                .unwrap()
                .unwrap()
        };
        let mut step = tf::SessionRunArgs::new();
        step.add_feed(&x, 0, &xs);
        step.add_target(&y);
        let y_hat_tok = step.request_fetch(&y, 0);
        self.session().run(&mut step).unwrap();

        step.fetch(y_hat_tok).unwrap()
    }

    /// elems of `xs` should have len <= model.in_dim
    /// elems of `ys` should have len <= model.out_dim
    pub unsafe fn train(&self, xs: &[&[u8]], ys: &[&[f32]], n_epoch: usize) {
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
        for _ in 0..n_epoch {
            session.run(&mut step).unwrap();
        }
    }
}
