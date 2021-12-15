import tensorflow.keras as K
import argparse as aps
import sys


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
        choices=["test"]
    )

    parser.add_argument(
        "--model-name", "-n"
    )
    return parser


def new_model(in_dim: int, out_dim: int):
    input_layer = K.Input(name="input", shape=(in_dim,))
    layer1 = K.layers.Dense(in_dim // 64, activation="relu")(input_layer)
    layer2 = K.layers.Dense(128, activation="relu")(layer1)
    return layer2


def run(args):
    model_type_handlers = {
        "test": run_test,
    }
    if args.model_type in model_type_handlers:
        model_type_handlers[args.model_type](args)
    else:
        sys.exit(1)


def run_test(args):
    print("test")


if __name__ == "__main__":
    parser = new_parser()
    args = parser.parse_args()
    run(args)
