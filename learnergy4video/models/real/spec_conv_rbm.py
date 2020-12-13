"""Gaussian-based Convolutional Restricted Boltzmann Machine.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy4video.utils.logging as l
from learnergy4video.models.binary import ConvRBM


logger = l.get_logger(__name__)


class SpecConvRBM(ConvRBM):
    """A GaussianConvRBM class provides the basic implementation for
    Gaussian-based Convolutional Restricted Boltzmann Machines.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(self, visible_shape=(28, 28), filter_shape=(7, 7), n_filters=5, n_channels=1,
                 steps=1, learning_rate=0.1, momentum=0, decay=0, use_gpu=False):
        """Initialization method.

        Args:
            visible_shape (tuple): Shape of visible units.
            filter_shape (tuple): Shape of filters.
            n_filters (int): Number of filters.
            n_channels (int): Number of channels.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: SpecConvRBM -> GaussianConvRBM.')

        # Override its parent class
        super(SpecConvRBM, self).__init__(visible_shape, filter_shape, n_filters, n_channels,
                                              steps, learning_rate, momentum, decay, use_gpu)

        self.p = 0.0

        # Normalize the input
        self.normalize = True

        logger.info('Class overrided.')

    def hidden_sampling(self, v):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.conv2d(v, self.W, bias=self.b)

        # Calculate probabilities
        probs = F.relu6(activations).detach()

        # Sampling a dropout mask from Bernoulli's distribution
        mask = (torch.full((activations.size(0), activations.size(1), activations.size(2), activations.size(3)),
                           1 - self.p, dtype=torch.float, device=self.device)).bernoulli()

        probs = torch.mul(probs, mask)
        #probs = torch.clamp(F.relu(activations), 0, 1).detach()

        return probs, probs

    def visible_sampling(self, h):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.conv_transpose2d(h, self.W, bias=self.a)

        # Calculate probabilities
        if self.normalize:
            probs = activations.detach()
        else:
            probs = F.relu6(activations).detach()
            #probs = torch.clamp(F.relu(activations), 0, 6).detach()

        return probs, probs

    def fit(self, samples, batch_size=128, epochs=10):
        """Fits a new RBM model.

        Args:
            samples (torch.tensor): Normalized samples from the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        # Resetting epoch's MSE to zero
        mse, batch_mse = 0, 0

        #frames = samples.size(1)

        # Checking whether GPU is avaliable and if it should be used
        if self.device == 'cuda':
            # Applies the GPU usage to the data
            samples = samples.cuda()

        frames = samples.size(1) # considering only one channel (gray-scale)
        max_value = samples.max()

        for fr in range(1, frames):
            samples[:, 0, :, :] += samples[:, fr, :, :]

        samples[:, 0, :, :] /= (max_value*frames)

        # Flattening the samples' batch
        samples = samples[:, 0, :, :].reshape(
                    len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])

        if self.normalize:
            samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + 1e-6))#.detach()


        # Performs the Gibbs sampling procedure
        _, _, _, _, visible_states = self.gibbs_sampling(samples)
   
        # Detaching the visible states from GPU for further computation
        visible_states = visible_states.detach()

        # Calculates the loss for further gradients' computation
        cost = torch.mean(self.energy(samples)) - \
                        torch.mean(self.energy(visible_states))

        # Initializing the gradient
        self.optimizer.zero_grad()

        # Computing the gradients
        cost.backward()

        # Updating the parameters
        self.optimizer.step()

        # Calculating current's batch MSE
        batch_mse += torch.div(
                        torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()

        return batch_mse

    def reconstruct(self, x):
        """Performs a reconstruction pass over the data.
        Args:
            x (torch.utils.data.DataLoader -> batch): An input tensor for computing the reconstruction pass.
        Returns:
            A tensor containing the DBN's outputs.
       
        """
        frames = x.size(1) # considering only one channel (gray-scale)
        max_value = x.max()

        for fr in range(1, frames):
            x[:, 0, :, :] += x[:, fr, :, :]
        x /= (max_value*frames)

        # Flattening the samples' batch
        x = x[:, 0, :, :].reshape(
                    len(x), self.n_channels, self.visible_shape[0], self.visible_shape[1])

        if self.normalize:
            x = ((x - torch.mean(x, 0, True)) / (torch.std(x, 0, True) + 1e-6))


        # Performs the Gibbs sampling procedure
        _, _, _, _, visible_states = self.gibbs_sampling(x)
   
        # Detaching the visible states from GPU for further computation
        visible_states = visible_states.detach()

        return visible_states

    def forward(self, samples):
        """Performs a forward pass over the data.
        Args:
            samples (torch.Tensor): An input tensor for computing the forward pass.
        Returns:
            A tensor containing the CRBM's outputs.
       
        """

        #self.p = 0
        frames = samples.size(1)

        max_value = samples.max()

        for fr in range(1, frames):
            samples[:, 0, :, :] += samples[:, fr, :, :]
        samples[:, 0, :, :] /= (max_value*frames)

        # Flattening the samples' batch
        samples = samples[:, 0, :, :].reshape(
                    len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])

        if self.normalize:
            samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + 1e-6))

        # Calculates the outputs of the model
        samples, _ = self.hidden_sampling(samples)

        return samples.detach()



