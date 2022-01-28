use libafl::inputs::HasBytesVec;
use tch;
use tch::{Device, IndexOp, Kind, Tensor};

/// create a tensor of input batch, input data will be copied
/// Aware that
///     1. oversize inputs will be truncated
///     2. short inputs will be pad with 0
pub unsafe fn input_batch_to_tensor(batch: &[impl HasBytesVec], in_dim: usize) -> Tensor {
    let ts = tch::Tensor::zeros(
        &[batch.len() as i64, in_dim as i64],
        (Kind::Uint8, Device::Cpu),
    );
    (0..batch.len() as i64).zip(batch).for_each(|(row, i)| {
        let bytes = i.bytes();
        let len = in_dim.min(bytes.len()) as i64;
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

/// tensor operators
pub mod ops {
    use super::{Kind, Tensor};
    pub fn standardize(ts: &Tensor) -> Tensor {
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
        let ts = unsafe { super::input_batch_to_tensor(&batch, in_dim) };
        println!("{}", ts.to_string(100).unwrap());
    }
}
