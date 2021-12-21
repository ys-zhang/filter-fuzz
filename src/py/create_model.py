import argparse as aps
import sys
import os
import tensorflow as tf
import tensorflow.keras as K

MODEL_DIR = "models"


def new_parser():
    parser = aps.ArgumentParser(
        description="Create TF NN model for filter"
    )

    parser.add_argument(
        "--in-dim", "-i",
        type=int
    )

    parser.add_argument(
        "--out-dim", "-o",
        type=int
    )

    parser.add_argument(
        "--model-type", "-m",
        choices=["test", "dense", ]
    )

    parser.add_argument(
        "--model-name", "-n"
    )
    return parser


def new_dense_model(in_dim: int, out_dim: int) -> K.Model:
    inputs = K.Input(name="inputs", shape=(in_dim,))
    layer1 = K.layers.Dense(in_dim // 64, activation="relu")(inputs)
    layer2 = K.layers.Dense(128, activation="relu")(layer1)
    outputs = K.layers.Dense(
        out_dim, activation="sigmoid", name="outputs")(layer2)
    model = K.Model(inputs=inputs, outputs=outputs)
    opt = K.optimizers.Adam(learning_rate=.0001)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['binary_accuracy', K.metrics.Recall(thresholds=.5)]
    )
    return model


def run(args):
    model_type_handlers = {
        "test": run_test,
        "dense": run_dense,
    }
    if args.model_type in model_type_handlers:
        model_type_handlers[args.model_type](args)
    else:
        sys.exit(1)


def run_test(_args):
    print("test")


def run_dense(args):
    in_dim, out_dim = args.in_dim, args.out_dim
    model_name = args.model_name
    model = new_dense_model(in_dim, out_dim)
    model_dir = os.path.join(MODEL_DIR, model_name)
    x_spec = tf.TensorSpec([None, in_dim], tf.float32, name="x")
    y_spec = tf.TensorSpec([None, out_dim], tf.float32, name="y")
    train_signature = model.make_train_function().get_concrete_function(x_spec, y_spec)
    model.save(model_dir, save_format='tf', signatures={"train": train_signature})


class Module:
    """
    This is only a helper class for saving Keras models to disk and load from
    rust tensorflow bindings.

    The reason of having this function is `Model.save` only saves the signature
    for prediction but have no support for training.

    for detail of Tensorflow SavedModel and tf.function, see:
        1. [Introduction to graphs and `tf.function`](https://www.tensorflow.org/guide/intro_to_graphs)
        2. [A tour of saved model signatures](https://blog.tensorflow.org/2021/03/a-tour-of-savedmodel-signatures.html)
        3. [Using the saved model format](https://www.tensorflow.org/guide/saved_model)
    """

    def __init__(self, model: K.Model):
        self._model = model

    @tf.function
    def train(self, x, y):
        # TODO: (yun)
        self._model.make_train_function

    @tf.function
    def predict(self, x):
        self._model.predict(x)


if __name__ == "__main__":
    parser = new_parser()
    args = parser.parse_args()
    run(args)
