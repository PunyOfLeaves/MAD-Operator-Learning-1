"""Plot the frozen computational boundary, without selecting solution cases."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def main(args):
    if args.output.exists():
        raise FileExistsError("Use a new output path to preserve previous figures")
    with np.load(args.geometry) as g:
        x = (g["nodes"] - g["center"]) / g["scale"]
        surface = g["surface"]
    fig = plt.figure(figsize=(8, 3.8))
    for i, azimuth in enumerate((-60, 120)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(x[surface], facecolor="#8dbac3",
                                           edgecolor="#344b50", linewidth=.18))
        for dim, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(x[:, dim].min() - .05, x[:, dim].max() + .05)
        ax.set_box_aspect(np.ptp(x, axis=0))
        ax.view_init(elev=22, azim=azimuth)
        ax.set_xlabel(r"$\xi_1$")
        ax.set_ylabel(r"$\xi_2$")
        ax.text2D(1.12, .60, r"$\xi_3$", transform=ax.transAxes, clip_on=False)
        ax.set_title(f"({chr(97+i)}) Boundary view {i+1}")
        ax.grid(False)
    fig.subplots_adjust(left=.02, right=.94, bottom=.1, top=.9, wspace=.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
