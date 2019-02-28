from argparse import ArgumentParser
from importlib import import_module

# Parse command line arguments
parser = ArgumentParser(description="Trains the neural nets")
parser.add_argument("--stencil", default=-1,
                    help="What stencil size to use for training InteriorNN")
args = parser.parse_args()

if args.stencil == -1:
    raise ValueError("You must provide a stencil size")

# Instantiate instance of given neural net class
NeuralNet = getattr(import_module("localnn"), "LocalNN")

# Train neural net
print(f"Training LocalNN with {args.stencil}x{args.stencil} stencil")
NeuralNet.train(int(args.stencil))
