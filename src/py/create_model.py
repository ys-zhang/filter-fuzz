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


class DenseModel(K.Model):
    """
    The reason of having this class is `keras.Model.save` only saves
    the signature for prediction, we also need the signature for
    training.

    for detail of Tensorflow SavedModel and tf.function, see:
        1. Introduction to graphs and `tf.function`
           (https://www.tensorflow.org/guide/intro_to_graphs)
        2. A tour of saved model signatures
           (https://blog.tensorflow.org/2021/03/a-tour-of-savedmodel-signatures.html)
        3. Using the saved model format
           (https://www.tensorflow.org/guide/saved_model)
    """
    def __init__(self, cmd_args):
        super().__init__()
        self.in_dim = cmd_args.in_dim
        self.out_dim = cmd_args.out_dim
        self.layer_1 = K.layers.Dense(self.in_dim // 64, activation="relu")
        self.layer_2 = K.layers.Dense(128, activation="relu")
        self.outputs_layer = K.layers.Dense(self.out_dim, activation="sigmoid")
        self.opt = K.optimizers.Adam(learning_rate=.0001)
        self.loss = K.losses.binary_crossentropy

    def call(self, x):
        t = self.layer_1(x)
        t = self.layer_2(t)
        y = self.outputs_layer(t)
        return y

    @tf.function
    def predict(self, x):
        return self(x, training=False)

    @tf.function
    def train(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self(x, training=True)
            loss = self.loss(y, y_hat)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    @property
    def signatures(self):
        x = tf.TensorSpec([None, self.in_dim], tf.float32, name="x")
        y = tf.TensorSpec([None, self.out_dim], tf.float32, name="y")
        return {
            "predict": self.predict.get_concrete_function(x),
            "train": self.train.get_concrete_function(x, y),
        }


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
    model = DenseModel(args)
    model_dir = os.path.join(MODEL_DIR, args.model_name)
    tf.saved_model.save(model, model_dir, signatures=model.signatures)


if __name__ == "__main__":
    parser = new_parser()
    args = parser.parse_args()
    run(args)
