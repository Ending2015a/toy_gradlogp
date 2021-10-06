# --- built in ---
import os

# --- 3rd party ---
import numpy as np
import torch
from torch import nn

# --- my module ---

__all__ = [
    'langevin_dynamics',
    'anneal_langevin_dynamics',
    'sample_score_field',
    'sample_energy_field'
]


# --- dynamics ---
def langevin_dynamics(
    score_fn,
    x,
    eps=0.1,
    n_steps=1000
):
    """Langevin dynamics

    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        x (torch.Tensor): input samples
        eps (float, optional): noise scale. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
    """
    for i in range(n_steps):
        x = x + eps/2. * score_fn(x).detach()
        x = x + torch.randn_like(x) * np.sqrt(eps)
    return x

def anneal_langevin_dynamics(
    score_fn,
    x,
    sigmas=None,
    eps=0.1,
    n_steps_each=100
):
    """Annealed Langevin dynamics

    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor, sigma: float) -> torch.Tensor
        x (torch.Tensor): input samples
        sigmas (torch.Tensor, optional): noise schedule. Defualts to None.
        eps (float, optional): noise scale. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
    """
    # default sigma schedule
    if sigmas is None:
        sigmas = np.exp(np.linspace(np.log(20), 0., 10))

    for sigma in sigmas:
        for i in range(n_steps_each):
            cur_eps = eps * (sigma / sigmas[-1]) ** 2
            x = x + cur_eps/2. * score_fn(x, sigma).detach()
            x = x + torch.randn_like(x) * np.sqrt(eps)
    return x


# --- sampling utils ---
def sample_score_field(
    score_fn,
    range_lim=4,
    grid_size=50,
    device='cpu'
):
    """Sampling score field from an energy model

    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        range_lim (int, optional): Range of x, y coordimates. Defaults to 4.
        grid_size (int, optional): Grid size. Defaults to 50.
        device (str, optional): torch device. Defaults to 'cpu'.
    """
    mesh = []
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    for i in x:
        for j in y:
            mesh.append(np.asarray([i, j]))
    mesh = np.stack(mesh, axis=0)
    x = torch.from_numpy(mesh).float()
    x = x.to(device=device)
    scores = score_fn(x.detach()).detach()
    scores = scores.cpu().numpy()
    return mesh, scores

def sample_energy_field(
    energy_fn,
    range_lim=4,
    grid_size=1000,
    device='cpu'
):
    """Sampling energy field from an energy model

    Args:
        energy_fn (callable): an energy function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        range_lim (int, optional): range of x, y coordinates. Defaults to 4.
        grid_size (int, optional): grid size. Defaults to 1000.
        device (str, optional): torch device. Defaults to 'cpu'.
    """
    energy = []
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    for i in y:
        mesh = []
        for j in x:
            mesh.append(np.asarray([j, i]))
        mesh = np.stack(mesh, axis=0)
        inputs = torch.from_numpy(mesh).float()
        inputs = inputs.to(device=device)
        e = energy_fn(inputs.detach()).detach()
        e = e.view(grid_size).cpu().numpy()
        energy.append(e)
    energy = np.stack(energy, axis=0) # (grid_size, grid_size)
    return energy