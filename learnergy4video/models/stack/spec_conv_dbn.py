"""Continuous-based Convolutional Deep Belief Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import learnergy4video.utils.logging as l
from learnergy4video.utils.collate import collate_fn
from learnergy4video.models.binary import ConvRBM
from learnergy4video.models.real import GaussianConvRBM, SpecConvRBM
from learnergy4video.core import Dataset, Model
import time

import os
workers = os.cpu_count()
if workers == None:
    workers = 0
else:
    workers -= 2

logger = l.get_logger(__name__)

MODELS = {
    'conv_rbm': ConvRBM,
    'cont_conv_rbm': GaussianConvRBM,
    'spec_conv_rbm': SpecConvRBM
}


class SpecCDBN(Model):
    """A Continuous ConvDBN class provides the basic implementation for
    Continuous-based input Convolutional DBNs.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(self, visible_shape=(28, 28), filter_shape=[(7, 7), (7, 7)], n_filters=[16, 16], n_channels=1,
                 n_layers=2, steps=1, learning_rate=(0.1,), momentum=(0,), decay=(0,), use_gpu=False):
        """Initialization method.

        Args:
            visible_shape (tuple): Shape of visible units.
            filter_shape (list of tuple): Shape of filters for each CRBM.
            n_filters (list of int): Number of filters for each CRBM.
            n_channels (int): Number of channels.
            n_layers (int): Number of layers
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: Model -> SpecCDBN.')

        # Override its parent class
        super(SpecCDBN, self).__init__(use_gpu=use_gpu)

        # Normalize the input
        self.normalize = True

        # Max Pooling layer
        #self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Shape of visible units
        self.visible_shape = visible_shape

        # Shape of filters
        self.filter_shape = filter_shape

        # Number of filters
        self.n_filters = n_filters

        # Number of channels
        self.n_channels = n_channels

        # Number of layers
        self.n_layers = n_layers

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # List of models (RBMs)
        self.models = []

        # For every possible layer
        for i in range(self.n_layers):
            if i == 0: model = 'spec_conv_rbm'
            else: model = 'cont_conv_rbm'

            # Shape of hidden units
            self.hidden_shape = (
                visible_shape[0] - filter_shape[i][0] + 1,
                visible_shape[1] - filter_shape[i][1] + 1)

            # Creates an CRBM
            m = MODELS[model](visible_shape=visible_shape, filter_shape=filter_shape[i], n_filters=n_filters[i],
                              n_channels=n_channels, steps=1, learning_rate=learning_rate[i],
                              momentum=momentum[i], decay=decay[i], use_gpu=use_gpu)

            # The new visible input stands for the hidden output incoming from the previous RBM
            #visible_shape = ( (m.hidden_shape[0]//2) + 1, (m.hidden_shape[1]//2) + 1) # FOR MAX-POOLING (2, 2)
            visible_shape = (visible_shape[0] - filter_shape[i][0] + 1, visible_shape[1] - filter_shape[i][1] + 1)
            n_channels = n_filters[i]

            # Appends the model to the list
            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')


    def fit(self, dataset, batch_size=128, epochs=(10, 10)):
        """Fits a new CDBN model.
        Args:
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (tuple): Number of training epochs per layer.
        Returns:
            MSE (mean squared error) from the training step.
        """


        # Initializing MSE and pseudo-likelihood as lists
        mse, pl = [], []

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate_fn)
        d = dataset

        # For every possible model (CRBM)
        for i, model in enumerate(self.models):
            logger.info('Fitting layer %d/%d ...', i + 1, self.n_layers)

            if i > 0:
                model.p = 0.0
                self.models[i].normalize = False
                model.normalize = False

            if i == 0:
            #if i < 1:
                try:
                    model = torch.load("spec_cdbn_model.pth")
                    #model = torch.load("1spec_cdbn0.pth")
                    model.eval()
                    model.cuda()
                    self.models[0] = model
                    #self.models[0] = model.models[0]
                    #self.models[1] = model.models[1]
                    continue
                except:
                    pass

            print('Model Normalization:', model.normalize)

            # For every epoch
            for epoch in range(epochs[i]):
                logger.info('Epoch %d/%d', epoch + 1, epochs[i])

                # Calculating the time of the epoch's starting
                start = time.time()

                model_mse, cont = 0, 0

                # For every batch
                for samples, _ in tqdm(batches):
                    #print("Samples", samples.size())
                    samples /= samples.max()
                    frames = samples.size(1)
                    mse_ = 0

                    if i == 0:
                        # Fits the first SpecCRBM
                        mse_ += model.fit(samples, len(samples), 1)
                    else:
                        for fr in range(1, frames):
                            samples[:, 0, :, :] += samples[:, fr, :, :]

                        samples[:, 0, :, :] /= frames
                        samples = samples[:, 0, :, :].reshape(
                                len(samples), self.n_channels, self.visible_shape[0], self.visible_shape[1])
                        samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + 1e-6))#.detach()

                        # Checking whether GPU is available and if it should be used
                        if self.device == 'cuda':
                            # Applies the GPU usage to the data
                            samples = samples.cuda()
   
                        for md in range(i):
                            samples, _ = self.models[md].hidden_sampling(samples)
                            #samples = self.max_pooling(samples)

                        mse_ += model.fit(samples, len(samples), 1)

                    model_mse += (mse_/frames)

                    if cont%50==0:
                        logger.info('MSE: %f', model_mse/(cont+1))
                    cont += 1
                model_mse /= len(batches)

                # Calculating the time of the epoch's ending
                end = time.time()

                # Detaches the variable from the computing graph
                samples = samples.detach()

                # Dumps the desired variables to the model's history
                model.dump(mse=model_mse.item(), time=end - start)

                # Appending the metrics
                mse.append(model_mse)

                logger.info('MSE: %f', model_mse)

            if i == 0:
                torch.save(model, "spec_cdbn_model.pth")

        return mse # , pl

    def reconstruct(self, x):
        """Performs a reconstruction pass over the data.
        Args:
            x (torch.utils.data.DataLoader -> batch): An input tensor for computing the reconstruction pass.
        Returns:
            A tensor containing the DBN's outputs.
       
        """

        neg_hidden_states = self.forward(x)

        for i, model in enumerate(reversed(self.models)):
            visible_probs, visible_states = model.visible_sampling(neg_hidden_states)
            neg_hidden_states = visible_probs

        neg_hidden_states.detach()

        return neg_hidden_states

    def forward(self, samples):
        """Performs a forward pass over the data.

        Args:
            samples (torch.Tensor): An input tensor for computing the forward pass.

        Returns:
            A tensor containing the Convolutional RBM's outputs.

        """

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

        # For every possible model
        for i, model in enumerate(self.models):
            # Calculates the outputs of the model
            samples, _ = model.hidden_sampling(samples)
        samples = self.max_pooling(samples)

        return samples.detach()
