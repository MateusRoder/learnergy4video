import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from torch.utils.data import DataLoader
from torch.fft import fftn, ifftn#, fftshift, ifftshift
from PIL import Image

from learnergy4video.visual.image import _rasterize
from learnergy4video.utils.collate import collate_fn
import learnergy4video.utils.constants as c
import learnergy4video.utils.exception as e
import learnergy4video.utils.logging as l
from learnergy4video.utils.fft_utils import fftshift, ifftshift

from learnergy4video.models.real.gaussian_rbm import GaussianRBM

import os
workers = os.cpu_count()
if workers == None:
    workers = 0
else:
    workers -= 2

logger = l.get_logger(__name__)


class FRRBM(GaussianRBM):
    """A Fourier-RBM class provides the basic implementation for Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    Note that this classes requires standardization of data as it uses variance equals to one throughout its learning procedure.
    This is a trick to ease the calculations of the hidden and visible layer samplings, as well as the cost function.

    References:

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

        logger.info('Overriding class: GaussianRBM -> FRRBM.')

        # Override its parent class
        super(FRRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        # Drop probability
        self.p = 0.0

        logger.info('Class overrided.')        

    
    def fit(self, dataset, batch_size=128, epochs=10, frames=6):
        """Fits a new RBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """
        
        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)

        # For every epoch
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Calculating the time of the epoch's starting
            start = time.time()

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse, pl, cst = 0, 0, 0

            # For every batch
            inner = tqdm.tqdm(total=len(batches), desc='Batch', position=1)
            for ii, batch in enumerate(batches):
                samples, _ = batch

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    samples = samples.cuda()

                mse2, pl2, cst2 = 0, 0, 0
                cost = 0

                # Initializing the gradient
                self.optimizer.zero_grad()

                for fr in range(frames):
                    #torch.autograd.set_detect_anomaly(True)                    
                    sps = samples[:, fr, :, :].squeeze()

                    # Creating the Fourier Spectrum
                    spec_data = fftshift(fftn(sps))[:,:,:,0]                    
                    spec_data = torch.abs(spec_data.squeeze())
                    
                    # Flattening the samples' batch
                    spec_data = spec_data.view(spec_data.size(0), self.n_visible)
                
                    # Normalizing the samples' batch
                    spec_data = ((spec_data - torch.mean(spec_data, 0, True)) / (torch.std(spec_data, 0, True) + c.EPSILON)).detach()

                    # Performs the Gibbs sampling procedure
                    _, _, _, _, visible_states = self.gibbs_sampling(spec_data)

                    # Calculates the loss for further gradients' computation
                    cost += torch.mean(self.energy(spec_data)) - \
                            torch.mean(self.energy(visible_states))

                    # Detaching the visible states from GPU for further computation
                    visible_states = visible_states.detach()

                    # Gathering the size of the batch
                    batch_size2 = sps.size(0)

                    # Calculating current's batch MSE
                    batch_mse = torch.div(
                        torch.sum(torch.pow(spec_data - visible_states, 2)), batch_size2).detach()

                    # Calculating the current's batch logarithm pseudo-likelihood
                    batch_pl = self.pseudo_likelihood(spec_data).detach()

                    # Summing up to epochs' MSE and pseudo-likelihood
                    mse2 += batch_mse
                    pl2 += batch_pl
                    cst2 += cost.detach()

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

                if ii % 100 == 99:
                    print('MSE:', (mse/ii).item(), 'Cost:', (cst/ii).item())
                    w8 = self.W.cpu().detach().numpy()
                    img = _rasterize(w8.T, img_shape=(72, 96), tile_shape=(30, 30), tile_spacing=(1, 1))
                    im = Image.fromarray(img)
                    im.save('w8_spec.png')

                    x = visible_states[:100].cpu().detach().reshape((100, 6912)).numpy()
                    x = _rasterize(x, img_shape=(72, 96), tile_shape=(10, 10), tile_spacing=(1, 1))
                    im = Image.fromarray(x)
                    im = im.convert("LA")
                    im.save('spectral.png')

                inner.update(1)

            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            pl /= len(batches)
            cst /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), pl=pl.item(), fe=cst.item(), time=end-start)

            logger.info(f'MSE: {mse} | log-PL: {pl} | Cost: {cst}')
        self.p = 0
        return mse, pl, cst

    def reconstruct(self, dataset, bs=2**7):
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
        batch_size = bs

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
        
        # For every batch
        inner = tqdm.tqdm(total=len(batches), desc='Batch', position=1)
        for _, batch in enumerate(batches):
            x, _ = batch
            frames = x.size(1) #frames            
            dy, dx = x.size(2), x.size(3)
            reconstructed = torch.zeros((bs, frames, self.n_visible))

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data            
                x = x.cuda()
                reconstructed = reconstructed.cuda()                

            for fr in range(frames):
                sps = x[:, fr, :, :].squeeze()

                # Creating the Fourier Spectrum
                spec_data = fftshift(fftn(sps))[:,:,:,0]
                spec_data = torch.abs(spec_data.squeeze())
                spec_data.detach()
                
                # Flattening the samples' batch
                spec_data = spec_data.view(spec_data.size(0),self.n_visible)

                # Normalizing the samples' batch
                spec_data = ((spec_data - torch.mean(spec_data, 0, True)) / (torch.std(spec_data, 0, True) + c.EPSILON)).detach()
            
                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(spec_data)

                visible_states = visible_states.detach()
                
                # Passing reconstructed data to a tensor
                reconstructed[:, fr, :] = visible_states

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(x.reshape((len(x), frames, dy*dx)) - visible_states, 2)), bs).detach()

            # Summing up the reconstruction's MSE
            mse += batch_mse

            inner.update(1)
            break

        # Normalizing the MSE with the number of batches
        mse /= len(batches)
        logger.info(f'MSE: {mse}')

        return mse, x, reconstructed

    def forward(self, x):
        """Performs a forward pass over the data.
        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.
        Returns:
            A tensor containing the DBN's outputs.
       
        """

        #self.p = 0
        frames = x.size(1) #frames            
        dy, dx = x.size(2), x.size(3)
        ds = torch.zeros((x.size(0), frames, self.n_hidden))

        # Checking whether GPU is avaliable and if it should be used
        if self.device == 'cuda':
            # Applies the GPU usage to the data            
            x = x.cuda()
            ds = ds.cuda()

        for fr in range(frames):
            sps = x[:, fr, :, :].squeeze()

            # Creating the Fourier Spectrum
            spec_data = fftshift(fftn(sps))[:,:,:,0]
            spec_data = torch.abs(spec_data.squeeze())
            
            # Flattening the samples' batch
            spec_data = spec_data.reshape(spec_data.size(0), self.n_visible)
        
            # Normalizing the samples' batch
            spec_data = ((spec_data - torch.mean(spec_data, 0, True)) / (torch.std(spec_data, 0, True) + c.EPSILON)).detach()

            spec_data, _ = self.hidden_sampling(spec_data)
            ds[:, fr, :] = spec_data.reshape((spec_data.size(0), self.n_hidden))

        x.detach()
        sps.detach()

        return ds.detach()
