# TODO

- [x] the overall skeleton for testing (forserver_simple)
- [x] tensorflow rust bindings seems not supporting Keras API very well. need some digging. Since this is a simple model, we may rewrite the model in tensorflow 1.X api. Refs:
    1. [rust tf example load (rust)](https://github.com/tensorflow/rust/blob/master/examples/regression_savedmodel.rs) and [rust tf example save (python)](https://github.com/tensorflow/rust/blob/master/examples/regression_savedmodel/regression_savedmodel.py)
    2. [Using the saved model format](https://www.tensorflow.org/guide/saved_model)
- [ ] migrate old python model
  - [ ] filter
  - [ ] train
  - [ ] test whether to run train on a batch of train samples
  - [x] deal with oversized inputs
  - [x] compress coverage map
  - [ ] test fitness of a model
  - [ ] change model

# Run

- run forkserver_simple:  `make forkserver_simple`.