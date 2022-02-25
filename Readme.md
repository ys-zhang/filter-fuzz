# TODO

- [x] the overall skeleton for testing (forserver_simple)
- [x] tensorflow rust bindings seems not supporting Keras API very well. need some digging. Since this is a simple model, we may rewrite the model in tensorflow 1.X api. Refs:
    1. [rust tf example load (rust)](https://github.com/tensorflow/rust/blob/master/examples/regression_savedmodel.rs) and [rust tf example save (python)](https://github.com/tensorflow/rust/blob/master/examples/regression_savedmodel/regression_savedmodel.py)
    2. [Using the saved model format](https://www.tensorflow.org/guide/saved_model)
- [ ] migrate old python model
  - [x] filter
  - [x] train
  - [x] test whether to run train on a batch of train samples
  - [x] deal with oversized inputs
  - [x] compress coverage map
  - [ ] test fitness of a model
  - [ ] change model
- [ ] migrate the libpng example from `libafl`

# Prepare

1. `llvm` version >= `llvm-11` is required by AFL++, which is used by the example to instrument targets
   and the `LLVM_CONFIG` variable needs to be correctly set in the Makefile, for detail see [issue](https://github.com/ys-zhang/filter-fuzz/issues/1) 

2. rust version should be edition 2021 as required in `Cargo.toml`. 
   
   rust can be installed by `$ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh` for detail see the [rust book](https://doc.rust-lang.org/book/ch01-01-installation.html)

   to update the stable rust toolchain run `rustup update stable`

# Run

- run forkserver_simple:  `make forkserver_simple`.