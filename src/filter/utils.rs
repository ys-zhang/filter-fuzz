pub mod model {
    // official tch examples see https://github.com/LaurentMazare/tch-rs
    use tch::{data::Iter2, nn, nn::OptimizerConfig, Reduction, Tensor};

    // the prediction model struct
    #[derive(Debug)]
    pub struct Model {
        module: Box<dyn nn::Module>,
        opt: nn::Optimizer,
        in_dim: usize,
        out_dim: usize,
    }

    impl Model {
        pub fn new_dense(vs: &nn::VarStore, in_dim: usize, out_dim: usize) -> Self {
            let ps = &vs.root() / "dense";
            let opt = nn::Adam::default()
                .build(vs, 0.0001)
                .expect("fail to build optimizer");
            let layer_1_out_dim = (in_dim as i64 / 64).max(128);
            let module = nn::seq()
                .add(nn::linear(
                    &ps / "dense-1",
                    in_dim as i64,
                    layer_1_out_dim,
                    Default::default(),
                ))
                .add(nn::func(|xs| xs.relu()))
                .add(nn::linear(
                    &ps / "dense-2",
                    layer_1_out_dim,
                    128,
                    Default::default(),
                ))
                .add(nn::func(|xs| xs.relu()))
                .add(nn::linear(
                    &ps / "output",
                    128,
                    out_dim as i64,
                    Default::default(),
                ))
                .add(nn::func(|xs| xs.sigmoid()));

            Self {
                module: Box::new(module),
                opt,
                in_dim,
                out_dim,
            }
        }
    }

    impl Model {
        pub fn in_dim(&self) -> usize {
            self.in_dim
        }

        pub fn out_dim(&self) -> usize {
            self.out_dim
        }

        pub fn train(&mut self, data: Iter2, epoch: usize) {
            // println!("Model::train start");
            let mut data = data;
            for _e in 0..epoch {
                for (xs, ys) in data.shuffle() {
                    let ys_hat = self.module.forward(&xs);
                    let loss = ys_hat.binary_cross_entropy::<Tensor>(&ys, None, Reduction::Mean);
                    self.opt.backward_step(&loss);
                }
            }
            // println!("Model::train end");
        }

        pub fn forward(&self, xs: &Tensor) -> Tensor {
            self.module.forward(xs)
        }
    }
}

pub mod data {
    use libafl::inputs::HasBytesVec;
    use std::ptr;
    use tch::{data::Iter2, Device, IndexOp, Kind, Tensor};

    /// create a tensor of input batch
    /// Aware that
    ///     1. input data will be copied
    ///     2. oversize inputs will be truncated
    ///     3. short inputs will be pad with 0
    ///     4. output is a float tensor
    ///     5. output tensor lives on CPU
    pub unsafe fn input_batch_to_tensor(batch: &[impl HasBytesVec], dim: usize) -> Tensor {
        let ts = tch::Tensor::zeros(
            &[batch.len() as i64, dim as i64],
            (Kind::Float, Device::Cpu),
        );
        (0..batch.len() as i64).zip(batch).for_each(|(row, i)| {
            let bytes = i.bytes();
            let len = dim.min(bytes.len()) as i64;
            // the `of_blob` creates a tensor without copy 
            // and do not assume ownership of the underlying buffer
            let ts_row = Tensor::of_blob(
                bytes.as_ptr(),
                &[len], // truncates the input to in_dim
                &[1],
                Kind::Uint8,
                Device::Cpu,
            );
            ts.i((row, ..len)).copy_(&ts_row);
        });
        ts
    }

    /// create a tensor for input, input data will be copied
    /// Aware that
    ///     1. input data will be copied
    ///     2. oversize inputs will be truncated
    ///     3. short inputs will be pad with 0
    ///     4. output is a byte tensor
    ///     5. output tensor on CPU
    #[allow(unused)]
    unsafe fn input_to_tensor(input: &impl HasBytesVec, dim: usize) -> Tensor {
        let bytes = input.bytes();
        // the `of_blob` function creates a tensor without copy
        // but the created tensor do not own the underlying memory buffer.
        // which is different from the `of_data` function that takes the ownership
        // see https://pytorch.org/cppdocs/api/function_namespacetorch_1ac009244049812a3efdf4605d19c5e79b.html
        let ts = Tensor::of_blob(
            bytes.as_ptr(),
            &[dim.min(bytes.len()) as i64], // truncates the input to in_dim
            &[1],
            Kind::Uint8,
            Device::Cpu,
        );
        if bytes.len() >= dim {
            ts.copy()
        } else {
            super::ops::pad(ts, (dim - bytes.len()) as i64, Kind::Uint8)
        }
    }

    pub struct SampleBuffer {
        size: usize,
        len: usize,
        x_dim: usize,
        y_dim: usize,
        x_buf: Vec<u8>,
        y_buf: Vec<u8>,
    }

    impl SampleBuffer {
        pub fn new(size: usize, x_dim: usize, y_dim: usize) -> Self {
            Self {
                x_buf: vec![0; x_dim * size],
                y_buf: vec![0; y_dim * size],
                x_dim,
                y_dim,
                size,
                len: 0,
            }
        }

        pub fn is_full(&self) -> bool {
            self.len == self.size
        }

        /// push(copy) x and y to the buffer
        /// slices that are too long will be truncated
        /// slices that are too short will be pad with 0
        /// return whether successfully pushed, fail of buffer is full
        pub fn push(&mut self, x: &[u8], y: &[u8]) -> bool {
            if self.is_full() {
                false
            } else {
                let x_cnt = x.len().min(self.x_dim);
                let y_cnt = y.len().min(self.y_dim);
                unsafe {
                    let x_dst = self
                        .x_buf
                        .as_mut_ptr()
                        .offset((self.len * self.x_dim) as isize);
                    let y_dst = self
                        .y_buf
                        .as_mut_ptr()
                        .offset((self.len * self.y_dim) as isize);
                    ptr::copy_nonoverlapping(x.as_ptr(), x_dst, x_cnt);
                    ptr::copy_nonoverlapping(y.as_ptr(), y_dst, y_cnt);
                }
                self.len += 1;
                true
            }
        }

        pub fn xs(&self) -> Tensor {
            unsafe {
                Tensor::of_blob(
                    self.x_buf.as_ptr(),
                    &[self.len as i64, self.x_dim as i64],
                    &[self.x_dim as i64, 1],
                    Kind::Uint8,
                    Device::Cpu,
                )
            }
        }

        pub fn ys(&self) -> Tensor {
            unsafe {
                Tensor::of_blob(
                    self.y_buf.as_ptr(),
                    &[self.len as i64, self.y_dim as i64],
                    &[self.y_dim as i64, 1],
                    Kind::Uint8,
                    Device::Cpu,
                )
            }
        }

        /// create a torch Iter2 from the Buffer
        pub fn iter2(&self, batch_size: usize) -> Iter2 {
            let xs = self.xs() + 0.0;
            let ys = self.ys() + 0.0;
            assert_eq!(ys.kind(), Kind::Float);
            assert_eq!(xs.kind(), Kind::Float);
            Iter2::new(&xs, &ys, batch_size as i64) // tensors are shallow cloned
        }

        pub fn to_iter2(self, batch_size: usize) -> Iter2 {
            self.iter2(batch_size)
        }

        pub fn truncate(&mut self) {
            self.x_buf.fill(0);
            self.y_buf.fill(0);
            self.len = 0;
        }
    }
}

/*
pub struct TrainBatch {
    xs: Tensor,
    ys: Tensor,
    len: i64,
    size: i64,
}

impl TrainBatch {
    /// create tensors on CPU to hold samples for training nn
    pub fn new(batch_size: usize, in_dim: usize, out_dim: usize) -> Self {
        Self {
            len: 0,
            size: batch_size as i64,
            xs: Tensor::zeros(
                &[batch_size as i64, in_dim as i64],
                (Kind::Float, Device::Cpu),
            ),
            ys: Tensor::zeros(
                &[batch_size as i64, out_dim as i64],
                (Kind::Float, Device::Cpu),
            ),
        }
    }

    pub fn is_full(&self) -> bool {
        self.len == self.size
    }

    pub fn add(&mut self, x: &Tensor, y: &Tensor) {
        if !self.is_full() {
            self.xs.i((self.len, ..)).copy_(x);
            self.ys.i((self.len, ..)).copy_(y);
            self.len += 1;
        }
    }

    pub fn truncate(&mut self) {
        self.len = 0;
    }
}

*/

/// tensor operators
pub mod ops {
    use tch::{Device, Kind, Tensor};

    /// standarize a 1-D or 2-D FLOAT tensor, if `ts` is a 2-D tensor
    /// then return a tensor with each row is the standardized version
    /// of the row from `ts`
    /// Notice:
    ///     1. input and output tensor are FLOAT tensors
    ///     2. input tensor must be a vector or matrix
    pub unsafe fn standardize(ts: &Tensor) -> Tensor {
        match ts.dim() {
            1 => {
                let mean = ts.mean(Kind::Float);
                let mut ct = ts - mean;
                let norm = ct.norm();
                ct /= norm;
                ct
            }
            2 => {
                /*
                  notice `ts.std_mean_dim(..)` calculate std and mean of tensor `ts`
                  but we need the std of `ts - mean_of_ts` here
                */
                let mean = ts.mean_dim(&[1], true, Kind::Float);
                let mut ct = ts - mean;
                let std = ct.std_dim(&[1], false, true);
                ct /= std;
                ct
            }
            _ => panic!("only support vectors and matrices"),
        }
    }

    /// padding a 1-D or 2-D tensor on the right
    /// Note that:
    ///     1. input tensor must be 1-D or 2-D
    ///     2. input and output tensors live on CPU
    #[allow(unused)]
    pub unsafe fn pad(ts: Tensor, right: i64, kind: Kind) -> Tensor {
        let ts_options = (kind, Device::Cpu);
        match ts.dim() {
            1 => Tensor::concat(&[ts, Tensor::zeros(&[right], ts_options)], 0),
            2 => Tensor::column_stack(&[
                &ts,
                &Tensor::zeros(&[ts.size()[0], right as i64], ts_options),
            ]),
            _ => panic!("pad only support 1-D and 2-D tensor"),
        }
    }
}

#[cfg(test)]
mod test {
    use libafl::inputs::HasBytesVec;

    struct Wrapper {
        data: Vec<u8>,
    }

    impl HasBytesVec for Wrapper {
        fn bytes(&self) -> &[u8] {
            &self.data
        }

        fn bytes_mut(&mut self) -> &mut Vec<u8> {
            &mut self.data
        }
    }

    #[test]
    fn input_batch_to_tensor() {
        let in_dim = 5;
        let batch: Vec<_> = (0..in_dim)
            .map(|i| Wrapper {
                data: vec![i as u8; i],
            })
            .collect();
        let ts = unsafe { super::data::input_batch_to_tensor(&batch, in_dim) };
        println!("{}", ts.to_string(100).unwrap());
    }
}
