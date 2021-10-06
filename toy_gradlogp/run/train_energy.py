"""
This example shows how to train an energy model on a Toy Dataset
"""
# --- built in ---
import os
import logging
import argparse
import functools
# --- 3rd party ---
import numpy as np
import torch
import tensorflow as tf
# disable GPU access from tensorflow
tf.config.set_visible_devices([], 'GPU')

import matplotlib.pyplot as plt

# --- my module ---
import toy_gradlogp as glogp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./log/')
    parser.add_argument(
        '--data',
        choices=['8gaussians', '2spirals', 'checkerboard', 'rings'],
        type=str, 
        default='2spirals',
        help='dataset'
    )
    parser.add_argument(
        '--loss',
        choices=['ssm-vr', 'ssm', 'deen', 'dsm'],
        type=str,
        default='ssm-vr',
        help='loss type'
    )
    parser.add_argument(
        '--noise',
        choices=['radermacher', 'sphere', 'gaussian'],
        type=str,
        default='gaussian',
        help='noise type'
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--size', type=int, default=1000000, help='dataset size')
    parser.add_argument('--eval_size', type=int, default=30000, help='dataset size for evaluation')
    parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
    parser.add_argument('--n_epochs', type=int, default=4, help='number of epochs to train')
    parser.add_argument('--n_slices', type=int, default=1, help='number of slices for sliced score matching')
    parser.add_argument('--n_steps', type=int, default=100, help='number of steps for langevin dynamics')
    parser.add_argument('--eps', type=float, default=0.01, help='noise scale for langevin dynamics')
    parser.add_argument('--gpu', action='store_true', default=False, help='enable gpu')
    parser.add_argument('--log_freq', type=int, default=1, help='logging frequency (unit: epoch)')
    parser.add_argument('--eval_freq', type=int, default=2, help='evaluation frequency (unit: epoch)')
    parser.add_argument('--vis_freq', type=int, default=2, help='visualization frequency (unit: epoch)')
    return parser.parse_args()

def main():
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    a = parse_args()
    # logging/assets path
    loss_type = a.loss
    log_root = os.path.join(a.logdir, a.loss, a.data)
    tb_path = log_root
    log_path = os.path.join(log_root, 'training.log')
    vis_path = os.path.join(log_root, 'vis/')
    save_path = os.path.join(log_root, 'save/weights.pt')
    # create datasets
    data = glogp.data.sample_2d(dataset=a.data, n_samples=a.size).numpy()
    train_dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(a.size)
    eval_data = glogp.data.sample_2d(dataset=a.data, n_samples=a.eval_size).numpy()
    eval_dataset = tf.data.Dataset.from_tensor_slices(eval_data)
    # create networks
    model = glogp.energy.Energy(
        net = glogp.energy.ToyMLP()
    )
    trainer = glogp.energy.Trainer(
        model,
        learning_rate = a.lr,
        n_slices = a.n_slices,
        loss_type = a.loss,
        noise_type = a.noise,
        device = 'cuda' if a.gpu else 'cpu'
    ).learn(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        n_epochs = a.n_epochs,
        batch_size = a.batch_size,
        log_freq = a.log_freq,
        eval_freq = a.eval_freq,
        vis_freq = a.vis_freq,
        vis_callback = functools.partial(
            visualize,
            vis_path = vis_path,
            data = data,
            steps = a.n_steps,
            eps = a.eps
        ),
        tb_logdir = tb_path
    )
    # save model
    trainer.model.save(save_path)

# --- plotting ---

def make_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_score_field(ax, trainer):
    mesh, scores = glogp.utils.sample_score_field(
        trainer.model.score,
        device=trainer.device
    )
    # draw scores
    ax.grid(False)
    ax.axis('off')
    glogp.vis.plot_scores(ax, mesh, scores)
    ax.set_title('Estimated scores', fontsize=16)

def plot_energy_field(ax, trainer):
    energy = glogp.utils.sample_energy_field(
        trainer.model,
        device=trainer.device
    )
    # draw energy
    ax.grid(False)
    ax.axis('off')
    glogp.vis.plot_energy(ax, energy)
    ax.set_title('Estimated energy', fontsize=16)

def plot_samples(ax, trainer, steps, eps):
    samples = []
    for i in range(1000):
        x = torch.rand(1000, 2) * 8 - 4
        x = x.to(device=trainer.device)
        x = glogp.utils.langevin_dynamics(
            trainer.model.score,
            x,
            n_steps=steps,
            eps=eps
        ).detach().cpu().numpy()
        samples.append(x)
    samples = np.concatenate(samples, axis=0)
    # draw energy
    ax.grid(False)
    ax.axis('off')
    glogp.vis.plot_data(ax, samples)
    ax.set_title('Sampled data', fontsize=16)

def visualize(trainer, vis_path, data, steps, eps):
    logging.info('Visualizing data ...')
    name = F"{trainer.num_epochs:04d}.png"
    vis_path = os.path.join(vis_path, name)
    make_path(vis_path)
    fig, axs = plt.subplots(figsize=(24, 6), ncols=4)
    # draw data samples
    axs[0].grid(False)
    axs[0].axis('off')
    glogp.vis.plot_data(axs[0], data)
    axs[0].set_title('Ground truth data', fontsize=16)
    plot_samples(axs[1], trainer, steps, eps)
    plot_energy_field(axs[2], trainer)
    plot_score_field(axs[3], trainer)
    for ax in axs:
        ax.set_box_aspect(1)
    plt.tight_layout()
    fig.savefig(vis_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close('all')

if __name__ == '__main__':
    main()