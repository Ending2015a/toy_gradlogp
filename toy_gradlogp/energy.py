# --- built in ---
import os
import sys
import time
import math
import logging
import functools

# --- 3rd party ---
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# --- my module ---

__all__ = [
    'Energy',
    'Trainer',
    'langevin_dynamics',
    'anneal_langevin_dynamics',
    'sample_score_field',
    'sample_energy_field'
]

# --- primitives ---
class Swish(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)

class ToyMLP(nn.Module):
    def __init__(
        self, 
        input_dim=2,
        output_dim=1,
        units=[300, 300],
        swish=True,
        dropout=False
    ):
        """Toy MLP from
        https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py#L198

        Args:
            input_dim (int, optional): input dimensions. Defaults to 2.
            output_dim (int, optional): output dimensions. Defaults to 1.
            units (list, optional): hidden units. Defaults to [300, 300].
            swish (bool, optional): use swish as activation function. Set False to use
                soft plus instead. Defaults to True.
            dropout (bool, optional): use dropout layers. Defaults to False.
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in units:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                Swish(out_dim) if swish == 'swish' else nn.Softplus(),
                nn.Dropout(.5) if dropout else nn.Identity()
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# --- energy model ---
class Energy(nn.Module):
    def __init__(self, net):
        """Energy model
        Args:
            net (nn.Module): An energy function, the output shape of
                the energy function should be (b, 1). The score is
                computed by grad(-E(x))
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logp = -self.net(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True)[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self

class Trainer():
    def __init__(
        self,
        model,
        learning_rate = 1e-3,
        clipnorm = 100.,
        n_slices = 1,
        loss_type = 'ssm-vr',
        noise_type = 'gaussian',
        device = 'cuda'
    ):
        """Energy based model trainer

        Args:
            model (nn.Module): energy-based model
            learning_rate (float, optional): learning rate. Defaults to 1e-4.
            clipnorm (float, optional): gradient clip. Defaults to 100..
            n_slices (int, optional): number of slices for sliced score matching loss.
                Defaults to 1.
            loss_type (str, optional): type of loss. Can be 'ssm-vr', 'ssm', 'deen',
                'dsm'. Defaults to 'ssm-vr'.
            noise_type (str, optional): type of noise. Can be 'radermacher', 'sphere'
                or 'gaussian'. Defaults to 'radermacher'.
            device (str, optional): torch device. Defaults to 'cuda'.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.n_slices = n_slices
        self.loss_type = loss_type.lower()
        self.noise_type = noise_type.lower()
        self.device = device

        self.model = self.model.to(device=self.device)
        # setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.num_gradsteps = 0
        self.num_epochs = 0
        self.progress = 0
        self.tb_writer = None

    def ssm_loss(self, x, v):
        """SSM loss from
        Sliced Score Matching: A Scalable Approach to Density and Score Estimation

        The loss is computed as
        s = -dE(x)/dx
        loss = vT*(ds/dx)*v + 1/2*(vT*s)^2

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises

        Returns:
            SSM loss
        """
        x = x.unsqueeze(0).expand(self.n_slices, *x.shape) # (n_slices, b, ...)
        x = x.contiguous().view(-1, *x.shape[2:]) # (n_slices*b, ...)
        x = x.requires_grad_()
        score = self.model.score(x) # (n_slices*b, ...)
        sv    = torch.sum(score * v) # ()
        loss1 = torch.sum(score * v, dim=-1) ** 2 * 0.5 # (n_slices*b,)
        gsv   = torch.autograd.grad(sv, x, create_graph=True)[0] # (n_slices*b, ...)
        loss2 = torch.sum(v * gsv, dim=-1) # (n_slices*b,)
        loss = (loss1 + loss2).mean() # ()
        return loss
    
    def ssm_vr_loss(self, x, v):
        """SSM-VR (variance reduction) loss from
        Sliced Score Matching: A Scalable Approach to Density and Score Estimation

        The loss is computed as
        s = -dE(x)/dx
        loss = vT*(ds/dx)*v + 1/2*||s||^2

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises

        Returns:
            SSM-VR loss
        """
        x = x.unsqueeze(0).expand(self.n_slices, *x.shape) # (n_slices, b, ...)
        x = x.contiguous().view(-1, *x.shape[2:]) # (n_slices*b, ...)
        x = x.requires_grad_()
        score = self.model.score(x) # (n_slices*b, ...)
        sv = torch.sum(score * v) # ()
        loss1 = torch.norm(score, dim=-1) ** 2 * 0.5 # (n_slices*b,)
        gsv = torch.autograd.grad(sv, x, create_graph=True)[0] # (n_slices*b, ...)
        loss2 = torch.sum(v*gsv, dim=-1) # (n_slices*b,)
        loss = (loss1 + loss2).mean() # ()
        return loss
    
    def deen_loss(self, x, v, sigma=0.1):
        """DEEN loss from
        Deep Energy Estimator Networks

        The loss is computed as
        
        x_ = x + v   # noisy samples
        s = dE(x_)/dx_
        loss = 1/2*||x - x_ + sigma^2*s||^2

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises
            sigma (int, optional): noise scale. Defaults to 1.

        Returns:
            DEEN loss
        """
        x = x.requires_grad_()
        v = v * sigma
        x_ = x + v
        s = sigma ** 2 * self.model.score(x_)
        loss = torch.norm(s+v, dim=-1)**2
        loss = loss.mean()/2.
        return loss

    def dsm_loss(self, x, v, sigma=0.1):
        """DSM loss from
        A Connection Between Score Matching
            and Denoising Autoencoders

        The loss is computed as


        Args:
            x ([type]): [description]
            v ([type]): [description]
            sigma (float, optional): [description]. Defaults to 0.1.
        """
        x = x.requires_grad_()
        v = v * sigma
        x_ = x + v
        s = self.model.score(x_)
        loss = torch.norm(s + v/(sigma**2), dim=-1)**2
        loss = loss.mean()/2.
        return loss

    def get_random_noise(self, x, n_slices=None):
        """Sampling random noises

        Args:
            x (torch.Tensor): input samples
            n_slices (int, optional): number of slices. Defaults to None.

        Returns:
            torch.Tensor: sampled noises
        """
        if n_slices is None:
            v = torch.randn_like(x, device=self.device)
        else:
            v = torch.randn((n_slices,)+x.shape, dtype=x.dtype, device=self.device)
            v = v.view(-1, *v.shape[2:]) # (n_slices*b, 2)

        if self.noise_type == 'radermacher':
            v = v.sign()
        elif self.noise_type == 'sphere':
            v = v/torch.norm(v, dim=-1, keepdim=True) * np.sqrt(v.shape[-1])
        elif self.noise_type == 'gaussian':
            pass
        else:
            raise NotImplementedError(
                f"Noise type '{self.noise_type}' not implemented."
            )
        return v
            
    
    def get_loss(self, x, v=None, sigma=0.08):
        """Compute loss

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor, optional): sampled noises. Defaults to None.

        Returns:
            loss
        """
        if self.loss_type == 'ssm-vr':
            v = self.get_random_noise(x, self.n_slices)
            loss = self.ssm_vr_loss(x, v)
        elif self.loss_type == 'ssm':
            v = self.get_random_noise(x, self.n_slices)
            loss = self.ssm_loss(x, v)
        elif self.loss_type == 'deen':
            v = self.get_random_noise(x, None)
            loss = self.deen_loss(x, v, sigma=sigma)
        elif self.loss_type == 'dsm':
            v = self.get_random_noise(x, None)
            loss = self.dsm_loss(x, v, sigma=sigma)
        else:
            raise NotImplementedError(
                f"Loss type '{self.loss_type}' not implemented."
            )

        return loss
    
    def train_step(self, batch, update=True):
        """Train one batch

        Args:
            batch (dict): batch data
            update (bool, optional): whether to update networks. 
                Defaults to True.

        Returns:
            loss
        """
        x = batch['samples']
        # move inputs to device
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        # compute losses
        loss = self.get_loss(x)
        # update model
        if update:
            # compute gradients
            loss.backward()
            # perform gradient updates
            grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def train(self, dataset, batch_size):
        """Train one epoch

        Args:
            dataset (tf.data.Dataset): Tensorflow dataset
            batch_size (int): batch size

        Returns:
            np.ndarray: mean loss
        """        
        all_losses = []
        dataset = dataset.batch(batch_size)
        for batch_data in dataset.as_numpy_iterator():
            sample_batch = {
                'samples': batch_data
            }
            loss = self.train_step(sample_batch)
            self.num_gradsteps += 1
            all_losses.append(loss)
        m_loss = np.mean(all_losses).astype(np.float32)
        return m_loss

    def eval(self, dataset, batch_size):
        """Eval one epoch

        Args:
            dataset (tf.data.Dataset): Tensorflow dataset
            batch_size (int): batch size

        Returns:
            np.ndarray: mean loss
        """        
        all_losses = []
        dataset = dataset.batch(batch_size)
        for batch_data in dataset.as_numpy_iterator():
            sample_batch = {
                'samples': batch_data
            }
            loss = self.train_step(sample_batch, update=False)
            all_losses.append(loss)
        m_loss = np.mean(all_losses).astype(np.float32)
        return m_loss

    def learn(
        self,
        train_dataset,
        eval_dataset = None,
        n_epochs = 5,
        batch_size = 100,
        log_freq = 1,
        eval_freq = 1,
        vis_freq = 1,
        vis_callback = None,
        tb_logdir = None
    ):
        """Train the model

        Args:
            train_dataset (tf.data.Dataset): training dataset
            eval_dataset (tf.data.Dataset, optional): evaluation dataset.
                Defaults to None.
            n_epochs (int, optional): number of epochs to train. Defaults to 5.
            batch_size (int, optional): batch size. Defaults to 100.
            log_freq (int, optional): logging frequency (epoch). Defaults to 1.
            eval_freq (int, optional): evaluation frequency (epoch). Defaults to 1.
            vis_freq (int, optional): visualizing frequency (epoch). Defaults to 1.
            vis_callback (callable, optional): visualization function. Defaults to None.
            tb_logdir (str, optional): path to tensorboard files. Defaults to None.

        Returns:
            self
        """        
        if tb_logdir is not None:
            self.tb_writer = SummaryWriter(tb_logdir)

        # initialize
        time_start = time.time()
        time_spent = 0
        total_epochs = n_epochs

        for epoch in range(n_epochs):
            self.num_epochs += 1
            self.progress = float(self.num_epochs) / float(n_epochs)
            # train one epoch
            loss = self.train(train_dataset, batch_size)
            # write tensorboard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f'train/loss', loss, self.num_epochs)
            
            if (log_freq is not None) and (self.num_epochs % log_freq == 0):
                logging.info(
                    f"[Epoch {self.num_epochs}/{total_epochs}]: loss: {loss}"
                )

            if (eval_dataset is not None) and (self.num_epochs % eval_freq == 0):
                # evaluate
                self.model.eval()
                eval_loss = self.eval(eval_dataset, batch_size)
                self.model.train()

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(f'eval/loss', eval_loss, self.num_epochs)
                
                logging.info(
                    f"[Eval {self.num_epochs}/{total_epochs}]: loss: {eval_loss}"
                )

            if (vis_callback is not None) and (self.num_epochs % vis_freq == 0):
                logging.debug("Visualizing")
                self.model.eval()
                vis_callback(self)
                self.model.train()
        return self


# --- dynamics ---
def langevin_dynamics(
    model,
    x,
    eps=0.1,
    n_steps=1000
):
    """Langevin dynamics

    Args:
        model (nn.Module): energy model
        x (torch.Tensor): input samples
        eps (float, optional): noise scale. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.

    Returns:
        torch.Tensor: sampled data
    """    
    for i in range(n_steps):
        x = x + eps/2. * model.score(x).detach()
        x = x + torch.randn_like(x) * np.sqrt(eps)
    return x

def anneal_langevin_dynamics(
    model,
    x,
    sigmas=None,
    eps=0.1,
    n_steps_each=100
):
    # default sigma schedule
    if sigmas is None:
        sigmas = np.exp(np.linspace(np.log(20), 0., 10))

    for sigma in sigmas:
        for i in range(n_steps_each):
            cur_eps = eps * (sigma / sigmas[-1]) ** 2
            x = x + cur_eps/2. * model.score(x, sigma).detach()
            x = x + torch.randn_like(x) * np.sqrt(eps)
    return x

# --- sampling utils ---
def sample_score_field(
    model,
    range_lim=4,
    grid_size=50,
    device='cpu'
):
    """Sampling score field from an energy model

    Args:
        model (nn.Module): energy model.
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
    scores = model.score(x.detach())
    scores = scores.detach().cpu().numpy()
    return mesh, scores

def sample_energy_field(
    model,
    range_lim=4,
    grid_size=1000,
    device='cpu'
):
    """Sampling energy field from an energy model

    Args:
        model (nn.Module): energy model.
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
        e = model(inputs.detach())
        e = e.detach().view(grid_size).cpu().numpy()
        energy.append(e)
    energy = np.stack(energy, axis=0) # (grid_size, grid_size)
    energy = energy[::-1] # flip y
    return energy