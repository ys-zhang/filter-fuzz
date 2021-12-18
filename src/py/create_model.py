import argparse as aps
import sys
import os
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
    inputs = K.Input(name="input", shape=(in_dim,), name="inputs")
    layer1 = K.layers.Dense(in_dim // 64, activation="relu")(inputs)
    layer2 = K.layers.Dense(128, activation="relu")(layer1)
    outputs = K.layers.Dense(out_dim, activation="sigmoid")(layer2, name="outputs")
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


def run_test(args):
    print("test")


def run_dense(args):
    in_dim, out_dim = args.in_dim, args.out_dim
    model_name = args.model_name
    model = new_dense_model(in_dim, out_dim)
    model.save(os.path.join(MODEL_DIR, model_name), save_format='tf')


if __name__ == "__main__":
    parser = new_parser()
    args = parser.parse_args()
    run(args)
