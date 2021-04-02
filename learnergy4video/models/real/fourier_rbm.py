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
    """A Multimodal Fourier-RBM class provides the basic implementation for Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    Note that this classes requires standardization of data as it uses variance equals to one throughout its learning procedure.
    This is a trick to ease the calculations of the hidden and visible layer samplings, as well as the cost function.

    References:

    """

    def __init__(self, n_visible=784, n_hidden=128, steps=1, learning_rate=0.1,
                 momentum=0, decay=0, temperature=1, use_gpu=True, mult=True):
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
            mult (boolean): To employ multimodal imput.

        """

        logger.info('Overriding class: GaussianRBM -> FRRBM.')

        # Amount of visible units (multimodal requires 2x the input size)
        self.n_visible = int(n_visible*2)

        # Override its parent class
        super(FRRBM, self).__init__(self.n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu, mult)

        # Multimodal input (2 w8 matrices) -> Default = False
        self.mult = mult

        # Drop probability
        self.p = 0.0

        logger.info('Class overrided.')        

    def visible_sampling(self, h, scale=False):
        """Performs the visible layer sampling, i.e., P(v|h).

        Args:
            h (torch.Tensor): A tensor incoming from the hidden layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the visible layer sampling.

        """

        # Calculating neurons' activations
        #activations = F.linear(h, self.c2*self.W, self.a)
        #activations1 = F.linear(h[:, :int(self.n_hidden/2)], self.c2*self.W[:int(self.n_visible/2), :int(self.n_hidden/2)],
        activations1 = F.linear(h[:, :int(self.n_hidden/2)], self.c2*self.W, self.a)
        activations2 = F.linear(h[:, int(self.n_hidden/2):], self.c2*self.W2, self.a2)
        activations = torch.cat((activations1, activations2), dim=-1)

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

    def hidden_sampling(self, v, scale=False):
        """Performs the hidden layer sampling, i.e., P(h|v).

        Args:
            v (torch.Tensor): A tensor incoming from the visible layer.
            scale (bool): A boolean to decide whether temperature should be used or not.

        Returns:
            The probabilities and states of the hidden layer sampling.

        """
        #print("tipo", self.W2.size(), self.b.size())
        # Calculating neurons' activations
        #activations1 = F.linear(v[:, :int(self.n_visible/2)], self.c1*self.W[:int(self.n_visible/2), :int(self.n_hidden/2)].t(), 
        activations1 = F.linear(v[:, :int(self.n_visible/2)], self.c1*self.W.t(), self.b)
        #self.b[:int(self.n_hidden/2)])
        
        #activations2 = F.linear(v[:, int(self.n_visible/2):], self.c1*self.W[int(self.n_visible/2):, int(self.n_hidden/2):].t(), 
        activations2 = F.linear(v[:, int(self.n_visible/2):], self.c1*self.W2.t(), self.b2)

        activations = torch.cat((activations1, activations2), dim=-1)

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

    def energy(self, v):
        """Calculates and frees the system's energy.

        Args:
            v (torch.Tensor): Samples to be energy-freed.

        Returns:
            The system's energy based on input samples.

        """

        # Calculate samples' activations
        activations1 = F.linear(v[:, :int(self.n_visible/2)], 
        self.c1*self.W.t(), self.b)
        #self.c1*self.W[:int(self.n_visible/2), :int(self.n_hidden/2)].t(), 
        #self.b[:int(self.n_hidden/2)])

        activations2 = F.linear(v[:, int(self.n_visible/2):], 
        self.c1*self.W2.t(), self.b2)
        #self.c1*self.W[int(self.n_visible/2):, int(self.n_hidden/2):].t(), 
        #self.b[int(self.n_hidden/2):])

        activations = torch.cat((activations1, activations2), dim=-1)

        # Creating a Softplus function for numerical stability
        s = nn.Softplus()

        # Calculate the hidden terms
        h = torch.sum(s(activations), dim=1)
        #h2 = torch.sum(s(activations2), dim=1)

        # Calculate the visible terms
        vv = torch.sum(0.5*((v[:, :int(self.n_visible/2)]-self.a)**2), dim=1)
        vv += torch.sum(0.5*((v[:, int(self.n_visible/2):]-self.a2)**2), dim=1)
        #vv1 = torch.sum(0.5*((v[:, :int(self.n_visible/2)]-
        #self.a[:int(self.n_visible/2)])**2), dim=1)
        #vv2 = torch.sum(0.5*((v[:, int(self.n_visible/2):]-
        #self.a[int(self.n_visible/2):])**2), dim=1)

        # Finally, gathers the system's energy
        energy = vv - h
        #energy1 = vv1 - h1
        #energy2 = vv2 - h2

        return energy

    
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
            #if e > 1:
            #    self.momentum = 0.9

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
                    #spec_data = 20*torch.log(torch.abs(spec_data.squeeze())+c.EPSILON)
                    spec_data = torch.abs(spec_data.squeeze())
                    spec_data.detach()
                    
                    # Flattening the samples' batch
                    sps = sps.view(sps.size(0), int(self.n_visible//2))
                    spec_data = spec_data.view(spec_data.size(0), int(self.n_visible//2))

                    # Concatenating the inputs
                    sps = torch.cat((sps, spec_data), dim=-1)
                
                    # Normalizing the samples' batch
                    sps = ((sps - torch.mean(sps, 0, True)) / (torch.std(sps, 0, True) + c.EPSILON)).detach()
                    #spec_data = ((spec_data - torch.mean(spec_data, 0, True)) / (torch.std(spec_data, 0, True) + c.EPSILON)).detach()                    
               
                    # Performs the Gibbs sampling procedure
                    _, _, _, _, visible_states = self.gibbs_sampling(sps)

                    # Calculates the loss for further gradients' computation
                    cost += torch.mean(self.energy(sps)) - \
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
                cost /= frames
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                mse2 /= frames
                pl2 /= frames
                cst2 /= frames
                #print('MSE:', (mse2).item(), 'Cost:', (cst2).item())

                mse += mse2
                pl  += pl2
                cst += cst2

                if ii % 100 == 99:
                    print('MSE:', (mse/ii).item(), 'Cost:', (cst/ii).item())
                    #w8 = self.W.cpu().detach().numpy()[1:, :]
                    w8 = self.W.cpu().detach().numpy()
                    #w8 = w8[:int(self.n_visible//2), :]
                    img = _rasterize(w8.T, img_shape=(72, 96), tile_shape=(30, 30), tile_spacing=(1, 1))
                    im = Image.fromarray(img)
                    im.save('w8_spec_ucf_.png')

                    w8 = self.W2.cpu().detach().numpy()
                    #w8 = w8[int(self.n_visible//2):, :]
                    img = _rasterize(w8.T, img_shape=(72, 96), tile_shape=(30, 30), tile_spacing=(1, 1))
                    im = Image.fromarray(img)
                    im.save('w8_spec.png')

                    x = visible_states[:100,:int(self.n_visible/2)].cpu().detach().reshape((100, 6912)).numpy()
                    x = _rasterize(x, img_shape=(72, 96), tile_shape=(10, 10), tile_spacing=(1, 1))
                    im = Image.fromarray(x)
                    im = im.convert("LA")
                    im.save('sample.png')
                    x = visible_states[:100,int(self.n_visible/2):].cpu().detach().reshape((100, 6912)).numpy()
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

            for fr in range(1, frames):
                sps = x[:, fr, :, :].squeeze()

                # Creating the Fourier Spectrum
                spec_data = fftshift(fftn(sps))[:,:,:,0]
                #spec_data = 20*torch.log(torch.abs(spec_data.squeeze())+c.EPSILON)
                spec_data = torch.abs(spec_data.squeeze())
                spec_data.detach()
                
                # Flattening the samples' batch
                sps = sps.view(sps.size(0), int(self.n_visible//2))
                spec_data = spec_data.view(spec_data.size(0), int(self.n_visible//2))

                # Concatenating the inputs
                sps = torch.cat((sps, spec_data), dim=-1)
            
                # Normalizing the samples' batch
                sps = ((sps - torch.mean(sps, 0, True)) / (torch.std(sps, 0, True) + c.EPSILON)).detach()
            
                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(sps)

                visible_states = visible_states.detach()
                
                # Passing reconstructed data to a tensor
                reconstructed[:, fr, :] = visible_states

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(x - visible_states, 2)), bs).detach()

            # Summing up the reconstruction's MSE
            mse += batch_mse

            inner.update(1)
            break

        # Normalizing the MSE with the number of batches
        mse /= len(batches)
        #print("samples", len(reconstructed))
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
            spec_data.detach()
            
            # Flattening the samples' batch
            sps = sps.view(sps.size(0), int(self.n_visible//2))
            spec_data = spec_data.view(spec_data.size(0), int(self.n_visible//2))
        
            # Concatenating the inputs
            sps = torch.cat((sps, spec_data), dim=-1)

            # Normalizing the samples' batch
            sps = ((sps - torch.mean(sps, 0, True)) / (torch.std(sps, 0, True) + c.EPSILON)).detach()
            #spec_data = ((spec_data - torch.mean(spec_data, 0, True)) / (torch.std(spec_data, 0, True) + c.EPSILON)).detach()

            sps, _ = self.hidden_sampling(sps)
            ds[:, fr, :] = sps.reshape((sps.size(0), 1, dy*dx))

        x.detach()
        sps.detach()

        return ds.detach()