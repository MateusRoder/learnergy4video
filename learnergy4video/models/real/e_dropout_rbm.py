"""Bernoulli-Bernoulli Restricted Boltzmann Machines with Energy-based Dropout.
"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy4video.utils.exception as ex
import learnergy4video.utils.logging as l
import learnergy4video.utils.constants as c
from learnergy4video.models.real import GaussianRBM
from learnergy4video.utils.collate import collate_fn

import os
workers = os.cpu_count()
if workers == None:
    workers = 0
else:
    workers  -= 2

logger = l.get_logger(__name__)


class EDropoutRBM(GaussianRBM):
    """An EDropoutRBM class provides the basic implementation for
    Bernoulli-Bernoulli Restricted Boltzmann Machines along with a Energy-based Dropout regularization.

    References:
        M. Roder, G. H. de Rosa, A. L. D. Rossi, J. P. Papa.
        Energy-based Dropout in Restricted Boltzmann Machines: Why Do Not Go Random.
        IEEE TETCI (2020).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: GaussianRBM -> EDropoutRBM.')

        # Override its parent class
        super(EDropoutRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        # Initializes the Energy-based Dropout mask
        self.M = torch.Tensor()

        logger.info('Class overrided.')

    @property
    def M(self):
        """torch.Tensor: Energy-based Dropout mask.

        """

        return self._M

    @M.setter
    def M(self, M):
        if not isinstance(M, torch.Tensor):
            raise ex.TypeError('`M` should be a PyTorch tensor')

        self._M = M

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(v, self.c1*self.W.t(), self.b)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.mul(torch.sigmoid(
                torch.div(activations, self.T)), self.M)

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.mul(torch.sigmoid(activations), self.M)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def total_energy(self, h, v):
        """Calculates the total energy of the model.

        Args:
            h (torch.Tensor): Hidden sampling states.
            v (torch.Tensor): Visible sampling states.

        Returns:
            The total energy of the model.

        """

        # Calculates the energy of the hidden layer
        e_h = -torch.mv(h, self.b)

        # Calculates the energy of the visible layer
        e_v = -torch.mv(v, self.a)

        # Calculates the energy of the reconstruction
        e_rec = -torch.mean(torch.mm(v, torch.mm(self.W, h.t())), dim=1)

        # Calculates the total energy
        energy = torch.mean(e_h + e_v + e_rec)

        return energy

    def energy_dropout(self, e, p_prob, n_prob):
        """Performs the Energy-based Dropout over the model.

        Args:
            e (torch.Tensor): Model's total energy.
            p_prob (torch.Tensor): Positive phase hidden probabilities.
            n_prob (torch.Tensor): Negative phase hidden probabilities.

        """

        # Calculates the Importance Level
        I = torch.div(torch.div(n_prob, p_prob), torch.abs(e))

        # Normalizes the Importance Level
        I = torch.div(I, torch.max(I, 0)[0])

        # Samples a probability tensor
        p = torch.rand((I.size(0), I.size(1)), device=self.device)

        # Calculates the Energy-based Dropout mask
        self.M = (I < p).float()

    def fit(self, dataset, batch_size=128, epochs=1, frames=6):
        """Fits a new E-drop RBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.
            frames (int): Number of frames.

        Returns:
            MSE (mean squared error) and(or not) log pseudo-likelihood from the training step.

        """

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate_fn)

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse, pl, cst = 0, 0, 0
            #ii = 1

            # For every batch
            for samples, y in tqdm(batches):
                
                cost = 0
                if self.device == 'cuda':
                    samples = samples.cuda()

                mse2, pl2, cst2 = 0, 0, 0
                
                for fr in range(frames):
                    # Returns the Energy-based Dropout mask to one
                    self.M = torch.ones(
                            (batch_size, self.n_hidden), device=self.device)

                    #torch.autograd.set_detect_anomaly(True)
                    # Flattening the samples' batch
                    sps = samples[:, fr, :, :]
                    sps = sps.view(sps.size(0), self.n_visible)

                    # Normalizing the samples' batch
                    sps = ((sps - torch.mean(sps, 0, True)) / (torch.std(sps, 0, True) + c.EPSILON)).detach()

                    # Performs the initial Gibbs sampling procedure (pre-dropout)
                    pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states = \
                        self.gibbs_sampling(sps)

                    # Calculating energy of positive phase sampling
                    e = self.total_energy(pos_hidden_states, sps)

                    # Calculating energy of negative phase sampling
                    e1 = self.total_energy(neg_hidden_states, visible_states)

                    # Performing the energy-based dropout
                    self.energy_dropout(e1 - e, pos_hidden_probs, neg_hidden_probs)

                    # Performs the post Gibbs sampling procedure (post-dropout)
                    _, _, _, _, visible_states = self.gibbs_sampling(sps)

                    # Detaching the visible states from GPU for further computation
                    visible_states = visible_states.detach()

                    # Calculates the loss for further gradients' computation
                    cost += torch.mean(self.energy(sps)) - \
                        torch.mean(self.energy(visible_states))

                    # Calculating current's batch MSE
                    batch_mse = torch.div(
                        torch.sum(torch.pow(sps - visible_states, 2)), batch_size)

                    # Calculating the current's batch logarithm pseudo-likelihood
                    batch_pl = self.pseudo_likelihood(sps).detach()

                    # Summing up to epochs' MSE and pseudo-likelihood
                    mse2 += batch_mse
                    pl2 += batch_pl
                    cst2 += cost.detach()

                # Initializing the gradient
                self.optimizer.zero_grad()
                
                # Computing the gradients
                cost /= frames
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                mse2 /= frames
                pl2 /= frames
                cst2 /= frames

                mse += mse2
                pl  += pl2
                cst += cst2
                
            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            cst /= len(batches)
            pl /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            #self.dump(mse=mse.item(), pl=pl.item(), time=end-start)
            self.dump(mse=mse.item(), pl=pl.item(), fe=cst.item(), time=end-start)

            #logger.info('MSE: %f | log-PL: %f', mse, pl)
            logger.info(f'MSE: {mse} | Cost: {cst}')

        return mse, pl, cst

class EDropoutRBM_Inner(EDropoutRBM):
    """An EDropoutRBM class provides the basic implementation for
    Bernoulli-Bernoulli Restricted Boltzmann Machines along with a Energy-based Dropout regularization, 
    for inner hidden layers.

    References:
        M. Roder, G. H. de Rosa, A. L. D. Rossi, J. P. Papa.
        Energy-based Dropout in Restricted Boltzmann Machines: Why Do Not Go Random.
        IEEE TETCI (2020).

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=False):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: EDropoutRBM -> EDropoutRBM_Inner.')

        # Override its parent class
        super(EDropoutRBM_Inner, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        logger.info('Class overrided.')

    def fit(self, samples, batch_size=128, epochs=1, frames=6):
        """Fits a new E-drop RBM model.

        Args:
            samples (torch.tensor): Samples containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.
            frames (int): Number of frames.

        Returns:
            MSE (mean squared error) and(or not) log pseudo-likelihood from the training step.

        """

        # Transforming the dataset into training batches
        batches = DataLoader(samples, batch_size=batch_size, shuffle=True, num_workers=workers)

        # For every epoch
        for epoch in range(epochs):
            
            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse = 0
            pl = 0
            cst = 0

            # For every batch
            for samples, _ in tqdm(batches):
                # Gathering the size of the batch
                batch_size = samples.size(0)
                
                for fr in range(frames):
                    # Returns the Energy-based Dropout mask to one
                    self.M = torch.ones(
                            (batch_size, self.n_hidden), device=self.device)

                    #torch.autograd.set_detect_anomaly(True)
                    # Flattening the samples' batch
                    sps = samples[:, fr, :, :]
                    sps = sps.view(sps.size(0), self.n_visible)

                    # Normalizing the samples' batch
                    sps = ((sps - torch.mean(sps, 0, True)) / (torch.std(sps, 0, True) + c.EPSILON)).detach()

                    # Performs the initial Gibbs sampling procedure (pre-dropout)
                    pos_hidden_probs, pos_hidden_states, neg_hidden_probs, neg_hidden_states, visible_states = \
                        self.gibbs_sampling(sps)

                    # Calculating energy of positive phase sampling
                    e = self.total_energy(pos_hidden_states, sps)

                    # Calculating energy of negative phase sampling
                    e1 = self.total_energy(neg_hidden_states, visible_states)

                    # Performing the energy-based dropout
                    self.energy_dropout(e1 - e, pos_hidden_probs, neg_hidden_probs)

                    # Performs the post Gibbs sampling procedure (post-dropout)
                    _, _, _, _, visible_states = self.gibbs_sampling(sps)

                    # Detaching the visible states from GPU for further computation
                    visible_states = visible_states.detach()

                    # Calculates the loss for further gradients' computation
                    cost = torch.mean(self.energy(sps)) - \
                        torch.mean(self.energy(visible_states))

                    # Initializing the gradient
                    self.optimizer.zero_grad()

                    # Computing the gradients
                    cost.backward()

                    # Updating the parameters
                    self.optimizer.step()

                    # Calculating current's batch MSE
                    batch_mse = torch.div(
                        torch.sum(torch.pow(sps - visible_states, 2)), batch_size).detach()

                    # Calculating the current's batch logarithm pseudo-likelihood
                    batch_pl = self.pseudo_likelihood(samples).detach()

                    # Summing up to epochs' MSE and pseudo-likelihood
                    mse += batch_mse
                    pl += batch_pl
                    cst += cost.detach()

                mse /= frames
                pl /= frames
                cst /= frames

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), fe=cst.item(), pl=pl.item(), time=end-start)
            #self.dump(mse=mse.item(), time=end-start)

            logger.info('MSE: %f | log-PL: %f', mse, pl)
            
        return mse, pl, cst.item()