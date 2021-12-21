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
    'ToyMLP',
    'Energy',
    'Trainer',
]

# --- primitives ---
class Swish(nn.Module):
    def __init__(self, dim=-1):
        """Swish activ bootleg from
        https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299

        Args:
            dim (int, optional): input/output dimension. Defaults to -1.
        """
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
                Swish(out_dim) if swish else nn.Softplus(),
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
        """A simple energy model

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
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
        s = -dE(x_)/dx_
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
        x_ = x + v   # noisy samples
        s = -dE(x_)/dx_
        loss = 1/2*||s + (x-x_)/sigma^2||^2

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises
            sigma (float, optional): noise scale. Defaults to 0.1.
        
        Returns:
            DSM loss
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
            
    
    def get_loss(self, x, v=None):
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
            loss = self.deen_loss(x, v)
        elif self.loss_type == 'dsm':
            v = self.get_random_noise(x, None)
            loss = self.dsm_loss(x, v)
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
