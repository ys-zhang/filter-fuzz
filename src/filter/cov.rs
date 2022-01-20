use super::{tf, utils, Debug, Filter};
use libafl::{
    inputs::HasBytesVec,
    observers::{map::MapObserver, ObserversTuple},
};
use num_traits::{cast::cast, PrimInt};
use std::marker::PhantomData;

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

impl<I, O, T> CovFilter<I, O, T>
where
    I: HasBytesVec,
    O: MapObserver<T>,
    T: tf::TensorType + PrimInt + Clone + Debug,
{
    fn filter(&self, ys: &tf::Tensor<f32>) -> Vec<f32> {
        // (num_of_sample, sample_dim)
        let (n, m) = {
            let shape = &ys.shape();
            (shape[0].unwrap() as usize, shape[1].unwrap() as usize)
        };

        todo!("yun")
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
        let prob = self.filter(&ys);
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
