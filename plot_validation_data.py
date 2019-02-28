from argparse import ArgumentParser
from localnn import LocalNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_style('whitegrid', {'font.sans-serif': "Helvetica"})
sns.set_palette(sns.color_palette('Set1'))

# Parse command line arguments
parser = ArgumentParser(description="Trains the neural nets")
parser.add_argument("--stencil", default=-1, type=int,
                    help="What stencil size to use for training LocalNN")
parser.add_argument("--num_points", default=10000, type=int,
                    help="Number of points to plot (default: 10000)")
args = parser.parse_args()

if args.stencil == -1:
    raise ValueError("You must provide a stencil size")

# Load training history file
# Load validation data
validation_data = np.load(f"training_data/localnn_{args.stencil}_validation_data.npz")
val_in, val_out = validation_data["val_in"], validation_data["val_out"]

# Build neural net
net = LocalNN(500, args.stencil)

# Subset data
val_in = val_in[:args.num_points,:]
val_out = val_out[:args.num_points,:]

# Infer tendencies from validation data
infer_out = net.model.predict(val_in, batch_size=1)

# Plot scatter plots
fig, axes = plt.subplots(figsize=(5,5))
axes.scatter(val_out[:], infer_out[:], s=2.0, alpha=0.3)
axes.set_title(f"$\eta$ {args.stencil} stencil,\
               r={pearsonr(val_out[:,0], infer_out[:,0])[0]:.2f}")

axes.set_xlabel("Actual")
axes.set_ylabel("Inferred")

# Make axis limits equal
x_min, x_max = axes.get_xlim()
y_min, y_max = axes.get_ylim()

axes.set_xlim([min(x_min, y_min), max(x_max, y_max)])
axes.set_ylim([min(x_min, y_min), max(x_max, y_max)])

# Save to file
plt.savefig(f"plots/localnn_{args.stencil}_corr.pdf", bbox_inches="tight")

plt.show()
