# --- built in ---
import os
import sys
import time
import math

# --- 3rd party ---
import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

# --- my module ---

__all__ = [
    'plot_data',
    'plot_scores',
    'plot_energy'
]

def plot_data(
    ax,
    data,
    range_lim=4,
    bins=1000,
    cmap=plt.cm.viridis
):
    rng = [[-range_lim, range_lim], [-range_lim, range_lim]]
    ax.hist2d(data[:,0], data[:, 1], range=rng, bins=bins, cmap=plt.cm.viridis)

def plot_scores(
    ax,
    mesh,
    scores,
    width=0.002
):
    """Plot score field

    Args:
        ax (): canvas
        mesh (np.ndarray): mesh grid
        scores (np.ndarray): scores
        width (float, optional): vector width. Defaults to 0.002
    """
    ax.quiver(mesh[:, 0], mesh[:, 1], scores[:, 0], scores[:, 1], width=width)

def plot_energy(
    ax,
    energy,
    cmap=plt.cm.viridis,
    flip_y=True
):
    if flip_y:
        energy = energy[::-1] # flip y
    ax.imshow(energy, cmap=cmap)