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
