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
        "--map-size",
        type=int,
        default=65536
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

    for detail of Tensorflow SavedModel and `tf.function`, see:
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
        self.map_size = cmd_args.map_size

        self.layer_1 = K.layers.Dense(self.in_dim // 64, activation="relu")
        self.layer_2 = K.layers.Dense(128, activation="relu")
        self.outputs_layer = K.layers.Dense(self.out_dim, activation="sigmoid")
        self.opt = K.optimizers.Adam(learning_rate=.0001)
        self.loss = K.losses.binary_crossentropy

    def call(self, x, **kwargs):
        t = self.layer_1(x)
        t = self.layer_2(t)
        y = self.outputs_layer(t)
        return y

    @tf.function
    def predict(self, x):
        y_hat = self(x, training=False)
        # need to normalise y_hat
        y_hat_normed, _ = tf.linalg.normalize(y_hat, axis=0)
        return {"y_hat": y_hat, "y_hat_normed": y_hat_normed}

    @tf.function
    def train(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self(x, training=True)
            loss = self.loss(y, y_hat)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    @tf.function
    def train_compress(self, x, y, y_indices):
        y = tf.gather(y, y_indices, axis=1)
        return self.train(x, y)

    @property
    def signatures(self):
        x = tf.TensorSpec([None, self.in_dim], tf.uint8, name="x")
        y_compressed = tf.TensorSpec([None, self.out_dim], tf.float32, name="y")
        y = tf.TensorSpec([None, self.map_size], tf.float32, name="y")
        y_indices = tf.TensorSpec([self.out_dim], tf.int32, name="y_indices")
        return {
            "predict": self.predict.get_concrete_function(x),
            "train": self.train.get_concrete_function(x, y_compressed),
            "train_compress": self.train_compress.get_concrete_function(x, y, y_indices)
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
    arg_parser = new_parser()
    arguments = arg_parser.parse_args()
    run(arguments)
