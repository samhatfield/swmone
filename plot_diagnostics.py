from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', rc={'font.sans-serif': "Helvetica"})
sns.set_palette(sns.color_palette('Set1'))

# Parse command line arguments
parser = ArgumentParser(description="Trains the neural nets")
parser.add_argument("--stencil", default=-1,
                    help="What stencil size to use for training InteriorNN")
args = parser.parse_args()

if args.stencil == -1:
    raise ValueError("You must provide a stencil size")

# Load training history file
diagnostics = np.loadtxt(f"models/localnn_{args.stencil}_history.txt")

# Plot mean squared error and correlation
ax1 = plt.subplot(211)
ax1.plot(diagnostics[:,0], label="validation data")
ax1.plot(diagnostics[:,1], label="training data")
ax1.set_ylabel("Mean squared error")
ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(diagnostics[:,2], label="validation data")
ax2.plot(diagnostics[:,3], label="training data")
ax2.set_ylabel("Correlation")
ax2.set_xlabel("Epoch")
plt.legend()
plt.suptitle(f"{args.stencil} stencil")
plt.savefig(f"plots/local_{args.stencil}_diag.pdf", bbox_inches="tight")
plt.show()
