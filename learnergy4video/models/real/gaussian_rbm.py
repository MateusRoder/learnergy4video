import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image
from learnergy4video.visual.image import _rasterize

import learnergy4video.utils.constants as c
import learnergy4video.utils.exception as e
import learnergy4video.utils.logging as l
from learnergy4video.models.binary import RBM

logger = l.get_logger(__name__)


class GaussianRBM(RBM):
    """A GaussianRBM class provides the basic implementation for Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    Note that this classes requires standardization of data as it uses variance equals to one throughout its learning procedure.
    This is a trick to ease the calculations of the hidden and visible layer samplings, as well as the cost function.

    References:
        K. Cho, A. Ilin, T. Raiko. Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

    """

    def __init__(self, n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=True):
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

        logger.info('Overriding class: RBM -> GaussianRBM.')

        # Override its parent class
        super(GaussianRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        logger.info('Class overrided.')

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations = F.linear(samples, self.W.t(), self.b)
        nan_mask = torch.isnan(samples)
        if nan_mask.any():
            print("Samples", samples)
            #for i in range(samples.size(0)):
            #    print(samples[i,:])
            raise RuntimeError(f'Indexes: {nan_mask.nonzero()}')

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1) 

        # Calculate the visible term
        #v = torch.mv(samples, self.a)
        v = torch.sum(0.5*((samples-self.a)**2), dim=1) 

        # Finally, gathers the system's energy
        energy = v - h
        #print("E", energy.mean().item(), )

        return energy

    def visible_sampling(self, h, scale=False):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(h, self.c2*self.W, self.a)

        # If scaling is true
        if scale:
            # Scale with temperature
            activations = torch.div(activations, self.T)

        # If scaling is false
        else:
            # Gathers the states as usual
            states = activations
        proba = torch.sigmoid(activations)

        return proba, activations


    def fit(self, dataset, batch_size=128, epochs=10, frames=6):
        """Fits a new RBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """
        def collate_fn(batches):
            try:
                # remove audio from the batch
                batches = [(d[0], d[2]) for d in batches]
                return default_collate(batches)
            except:
                return default_collate(batches)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, collate_fn=collate_fn)
        
        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse, pl, cst = 0, 0, 0

            # For every batch
            #for samples, cc in batches:
            #inner = tqdm.tqdm(total=len(batches), desc='Batch', position=1)
            #for ii, batch in enumerate(batches):
            ii = 1
            for samples, y in tqdm(batches):
                #samples, y = batch
                #samples = torch.tensor(samples, requires_grad=False)
                cost = 0
                if self.device == 'cuda':
                    samples = samples.cuda()

                # Initializing the gradient
                self.optimizer.zero_grad()

                mse2, pl2, cst2 = 0, 0, 0

                for fr in range(frames):
                    torch.autograd.set_detect_anomaly(True)
                    # Flattening the samples' batch
                    sps = samples[:, fr, :, :]
                    sps = sps.view(sps.size(0), self.n_visible)

                    # Normalizing the samples' batch
                    sps = ((sps - torch.mean(sps, 0, True)) / (torch.std(sps, 0, True) + c.EPSILON)).detach()
                
                    # Checking whether GPU is avaliable and if it should be used
                    if self.device == 'cuda':
                    # Applies the GPU usage to the data
                        sps = sps.cuda()

                    # Performs the Gibbs sampling procedure
                    _, _, _, _, visible_states = self.gibbs_sampling(sps)

                    # Calculates the loss for further gradients' computation
                    cost = cost + torch.mean(self.energy(sps)) - \
                        torch.mean(self.energy(visible_states))

                    # Detaching the visible states from GPU for further computation
                    visible_states = visible_states.detach()

                    # Gathering the size of the batch
                    batch_size2 = sps.size(0)

                    # Calculating current's batch MSE
                    batch_mse = torch.div(
                        torch.sum(torch.pow(sps - visible_states, 2)), batch_size2).detach()

                    # Calculating the current's batch logarithm pseudo-likelihood
                    batch_pl = self.pseudo_likelihood(sps).detach()

                    # Summing up to epochs' MSE and pseudo-likelihood
                    mse2 += batch_mse
                    pl2 += batch_pl
                    cst2 += cost.detach()

                # Computing the gradients
                cost = cost/frames
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                mse2 /= frames
                pl2 /= frames
                cst2 /= frames

                mse += mse2
                pl  += pl2
                cst += cst2
                if ii % 100 == 99:
                    print('MSE:', (mse/ii).item(), 'Cost:', (cst/ii).item())
                    w8 = self.W.cpu().detach().numpy()
                    img = _rasterize(w8.T, img_shape=(72, 96), tile_shape=(30, 30), tile_spacing=(1, 1))
                    #img = _rasterize(w8.T, img_shape=(96, 128), tile_shape=(30, 30), tile_spacing=(1, 1))
                    im = Image.fromarray(img)
                    im.save('w8_pre_ucf101.png')

                ii += 1

            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            pl /= len(batches)
            cst /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), pl=pl.item(), fe=cst.item(), time=end-start)
            

            logger.info(f'MSE: {mse} | log-PL: {pl} | Cost: {cst}')

        return mse, pl, cst

    def reconstruct(self, dataset, bs):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info(f'Reconstructing new samples ...')

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        #batch_size = len(dataset)
        batch_size = bs

        def collate_fn(batches):
            try:
                # remove audio from the batch
                batches = [(d[0], d[2]) for d in batches]
                return default_collate(batches)
            except:
                return default_collate(batches)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        
        reconstructed = []
        original = []
        # For every batch
        inner = tqdm.tqdm(total=len(batches), desc='Batch', position=1)
        for _, batch in enumerate(batches):
            samples, y = batch
            samples = samples.view(samples.size(0)*samples.size(1), self.n_visible)

            #med = torch.mean(samples, 1).view(samples.size(0), 1)
            #sd = torch.std(samples, 1).view(samples.size(0), 1)
            #samples-=med
            #samples/=sd

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            _, pos_hidden_states = self.hidden_sampling(samples)

            # Calculating visible probabilities and states
            visible_probs, _ = self.visible_sampling(pos_hidden_states)

            samples = samples.detach()
            visible_probs = visible_probs.detach()
            pos_hidden_states = pos_hidden_states.detach()

            # Passing reconstructed data to a tensor
            reconstructed.append(visible_probs)
            original.append(samples)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_probs, 2)), batch_size).detach()

            # Summing up the reconstruction's MSE
            mse += batch_mse

            inner.update(1)
            break

        # Normalizing the MSE with the number of batches
        mse /= len(batches)
        #print("samples", len(reconstructed))
        logger.info(f'MSE: {mse}')

        return mse, samples, visible_probs


class VarianceGaussianRBM(RBM):
    """A VarianceGaussianRBM class provides the basic implementation for Gaussian-Bernoulli Restricted Boltzmann Machines (without standardization).

    Note that this class implements a new cost function that takes in account a new learning parameter: variance (sigma). Therefore,
    there is no need to standardize the data, as the variance will be trained throughout the learning procedure.

    References:
        K. Cho, A. Ilin, T. Raiko. Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
        International conference on artificial neural networks (2011).

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

        logger.info('Overriding class: RBM -> VarianceGaussianRBM.')

        # Override its parent class
        super(VarianceGaussianRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                                  momentum, decay, temperature, use_gpu)

        # Variance parameter
        self.sigma = nn.Parameter(torch.ones(n_visible))

        # Updating optimizer's parameters with `sigma`
        self.optimizer.add_param_group({'params': self.sigma})

        # Re-checks if current device is CUDA-based due to new parameter
        if self.device == 'cuda':
            # If yes, re-uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')

    @property
    def sigma(self):
        """torch.nn.Parameter: Variance parameter.

        """

        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if not isinstance(sigma, nn.Parameter):
            raise e.TypeError('`sigma` should be a PyTorch parameter')

        self._sigma = sigma

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(
            torch.div(v, torch.pow(self.sigma, 2)), self.W.t(), self.b)

        # If scaling is true
        if scale:
            # Calculate probabilities with temperature
            probs = torch.sigmoid(torch.div(activations, self.T))

        # If scaling is false
        else:
            # Calculate probabilities as usual
            probs = torch.sigmoid(activations)

        # Sampling current states
        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(self, h, scale=False):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        activations = F.linear(h, self.W, self.a)

        # Checks if device is CPU-based
        if self.device == 'cpu':
            # If yes, variance needs to have size equal to (batch_size, n_visible)
            sigma = torch.repeat_interleave(
                self.sigma, activations.size(0), dim=0)

        # If it is GPU-based
        else:
            # Variance needs to have size equal to (n_visible)
            sigma = self.sigma

        # Sampling current states from a Gaussian distribution
        states = torch.normal(activations, torch.pow(sigma, 2))

        return states, activations

    def energy(self, samples):
        """Calculates and frees the system's energy.

        Args:
            samples (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculating the potency of variance
        sigma = torch.pow(self.sigma, 2)

        # Calculate samples' activations
        activations = F.linear(torch.div(samples, sigma), self.W.t(), self.b)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden term
        h = torch.sum(s(activations), dim=1)

        # Calculate the visible term
        # Note that this might be improved
        v = torch.sum(
            torch.div(torch.pow(samples - self.a, 2), 2 * sigma), dim=1)

        # Finally, gathers the system's energy
        energy = -v - h

        return energy
