import time
import tqdm
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import learnergy4video.utils.exception as e
import learnergy4video.utils.logging as l
from learnergy4video.models.binary import RBM
from learnergy4video.utils.collate import collate_fn

#from PIL import Image
#from learnergy4video.visual.image import _rasterize

logger = l.get_logger(__name__)


class SigmoidRBM(RBM):
    """A SigmoidRBM class provides the basic implementation for Sigmoid-Bernoulli Restricted Boltzmann Machines.

    References:
        G. Hinton. A practical guide to training restricted Boltzmann machines.
        Neural networks: Tricks of the trade (2012).

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

        logger.info('Overriding class: RBM -> SigmoidRBM.')

        # Override its parent class
        super(SigmoidRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                         momentum, decay, temperature, use_gpu)

        logger.info('Class overrided.')

    def visible_sampling(self, h, scale=False):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The states and probabilities of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(h, self.c2*self.W, self.a)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(torch.div(activations, self.T))

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Copying states as current probabilities
        states = probs

        return states, probs

    def fit(self, sps, batch_size=128, epochs=1, frames=6):
        """Fits a new RBM model.

        Args:
            sps (torch.tensor): Sampels containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.
            frames (int): Number of frames.

        Returns:
            MSE (mean squared error), log pseudo-likelihood and free-energy from the training step.

        """
        
        # For every epoch
        for _ in range(epochs):
            #logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse, pl, cst = 0, 0, 0

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                sps = sps.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(sps)


                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(sps)) - \
                        torch.mean(self.energy(visible_states))

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size2 = sps.size(0)

                # Calculating current's batch MSE
                batch_mse = torch.div(
                        torch.sum(torch.pow(sps - visible_states, 2)), batch_size2).detach()

                # Calculating the current's batch logarithm pseudo-likelihood
                batch_pl = self.pseudo_likelihood(sps).detach()

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Summing up to epochs' MSE and pseudo-likelihood
                mse += batch_mse
                pl += batch_pl
                cst += cost.detach()

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), pl=pl.item(), fe=cst.item(), time=end-start)

            #logger.info(f'MSE: {mse} | log-PL: {pl}')

        return mse, pl, cst
