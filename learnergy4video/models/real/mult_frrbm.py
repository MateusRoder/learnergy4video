import os
import time
import torch
import tqdm

from PIL import Image

from torch.utils.data import DataLoader
from torch.fft import fftn, ifftn

import learnergy4video.utils.constants as c
import learnergy4video.utils.exception as e
import learnergy4video.utils.logging as l

from learnergy4video.visual.image import _rasterize
from learnergy4video.utils.collate import collate_fn
from learnergy4video.core.dataset import Dataset
from learnergy4video.core.model import Model
from learnergy4video.models.real import GaussianRBM, FRRBM
from learnergy4video.utils.fft_utils import fftshift, ifftshift

workers = os.cpu_count()
if workers == None:
    workers = 0
else:
    workers -= 2

logger = l.get_logger(__name__)

class MultFRRBM(Model):
    """A  class provides the basic implementation for Deep Belief Networks.
    References:
        Roder 2021
    """

    def __init__(self, n_visible=(72, 96), n_hidden=(128, 128), steps=(1, 1), learning_rate=(0.0001, 0.0001), 
		 momentum=(0, 0), decay=(0, 0), temperature=(1, 1), use_gpu=True):
        """Initialization method.
        Args:
            n_visible (tuple): Input shape of visible units.
            n_hidden (tuple): Amount of hidden units per layer.
            steps (tuple): Number of Gibbs' sampling steps per layer.
            learning_rate (tuple): Learning rate per layer.
            momentum (tuple): Momentum parameter per layer.
            decay (tuple): Weight decay used for penalization per layer.
            temperature (tuple): Temperature factor per layer.
            use_gpu (boolean): Whether GPU should be used or not.
        """

        logger.info('Overriding class: Model -> MultFRRBM.')

        # Override its parent class
        super(MultFRRBM, self).__init__(use_gpu=use_gpu)

        # Number of Multimodal input
        self.n_layers = 2

        # Shape of visible input
        self.visible_shape = n_visible

        # Amount of visible units
        self.n_visible = int(n_visible[0]*n_visible[1])

        # Amount of hidden units per layer
        self.n_hidden = n_hidden

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # Temperature factor
        self.T = temperature

        # List of models (RBMs)
        self.models = [FRRBM(self.n_visible, self.n_hidden[0], self.steps[0], self.lr[0], self.momentum[0], self.decay[0], self.T[0], use_gpu),
		       GaussianRBM(self.n_visible, self.n_hidden[1], self.steps[1], self.lr[1], self.momentum[1], self.decay[1], self.T[1], use_gpu)]

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')

    @property
    def visible_shape(self):
        """int: Shape of visible layer.
        """

        return self._visible_shape

    @visible_shape.setter
    def visible_shape(self, visible_shape):
        if not isinstance(visible_shape, (list, tuple)):
            raise e.TypeError('`visible_shape` should be a list or tuple')

        self._visible_shape = visible_shape

    @property
    def n_visible(self):
        """int: Number of visible units.
        """

        return self._n_visible

    @n_visible.setter
    def n_visible(self, n_visible):
        if not isinstance(n_visible, int):
            raise e.TypeError('`n_visible` should be an integer')
        if n_visible <= 0:
            raise e.ValueError('`n_visible` should be > 0')

        self._n_visible = n_visible

    @property
    def n_hidden(self):
        """tuple: Tuple of hidden units.
        """

        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden):
        if not isinstance(n_hidden, (list, tuple)):
            raise e.TypeError('`n_hidden` should be a tuple')

        self._n_hidden = n_hidden

    @property
    def steps(self):
        """tuple: Number of steps Gibbs' sampling steps per layer.
        """

        return self._steps

    @steps.setter
    def steps(self, steps):
        if not isinstance(steps, tuple):
            raise e.TypeError('`steps` should be a tuple')
        if len(steps) != self.n_layers:
            raise e.SizeError(
                f'`steps` should have size equal as {self.n_layers}')

        self._steps = steps

    @property
    def lr(self):
        """tuple: Learning rate per layer.
        """

        return self._lr

    @lr.setter
    def lr(self, lr):
        if not isinstance(lr, tuple):
            raise e.TypeError('`lr` should be a tuple')
        if len(lr) != self.n_layers:
            raise e.SizeError(
                f'`lr` should have size equal as {self.n_layers}')

        self._lr = lr

    @property
    def momentum(self):
        """tuple: Momentum parameter per layer.
        """

        return self._momentum

    @momentum.setter
    def momentum(self, momentum):
        if not isinstance(momentum, tuple):
            raise e.TypeError('`momentum` should be a tuple')
        if len(momentum) != self.n_layers:
            raise e.SizeError(
                f'`momentum` should have size equal as {self.n_layers}')

        self._momentum = momentum

    @property
    def decay(self):
        """tuple: Weight decay per layer.
        """

        return self._decay

    @decay.setter
    def decay(self, decay):
        if not isinstance(decay, tuple):
            raise e.TypeError('`decay` should be a tuple')
        if len(decay) != self.n_layers:
            raise e.SizeError(
                f'`decay` should have size equal as {self.n_layers}')

        self._decay = decay

    @property
    def T(self):
        """tuple: Temperature factor per layer.
        """

        return self._T

    @T.setter
    def T(self, T):
        if not isinstance(T, tuple):
            raise e.TypeError('`T` should be a tuple')
        if len(T) != self.n_layers:
            raise e.SizeError(f'`T` should have size equal as {self.n_layers}')

        self._T = T

    @property
    def models(self):
        """list: List of models (RBMs).
        """

        return self._models

    @models.setter
    def models(self, models):
        if not isinstance(models, list):
            raise e.TypeError('`models` should be a list')

        self._models = models

    def fit(self, dataset, batch_size=128, epochs=10, frames=6):
        """Fits a new MultFRRBM model.
        Args:
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (list): Number of training epochs per layer.
        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.
        """
            
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)

        for ep in range(epochs):
            logger.info(f'Epoch {ep+1}/{epochs}')

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse, pl, cst = 0, 0, 0

            inner_trans = tqdm.tqdm(total=len(batches), desc='Batch', position=1)
            start = time.time()

            for ii, batch in enumerate(batches):
                x, y = batch

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    x = x.cuda()

                mse2, pl2, cst2 = 0, 0, 0
                cost, cost2 = 0, 0

                # Initializing the gradient
                #self.models[1].optimizer.zero_grad()

                for fr in range(frames):
                    x_ = x[:, fr, :, :].squeeze()

                    spec_data = fftshift(fftn(x_))[:,:,:,0]                    
                    spec_data = torch.abs(spec_data.squeeze())
                    spec_data = spec_data.reshape(spec_data.size(0), self.n_visible)
                    spec_data = ((spec_data - torch.mean(spec_data, 0, True)) / (torch.std(spec_data, 0, True) + c.EPSILON)).detach()

                    x_ = x_.reshape(x.size(0), self.n_visible)
                    x_ = ((x_ - torch.mean(x_, 0, True))/(torch.std(x_, 0, True) + c.EPSILON)).detach()

                    # Performs the Gibbs sampling procedure
                    _, _, _, _, visible_states = self.models[0].gibbs_sampling(spec_data)
                    _, _, _, _, visible_states2 = self.models[1].gibbs_sampling(x_)

                    # Calculates the loss for further gradients' computation
                    cost = torch.mean(self.models[0].energy(spec_data)) - torch.mean(self.models[0].energy(visible_states))
                    cost2 = torch.mean(self.models[1].energy(x_)) - torch.mean(self.models[1].energy(visible_states2))

                    # Initializing the gradient
                    self.models[0].optimizer.zero_grad()
                    self.models[1].optimizer.zero_grad()

                    # Computing the gradients
                    #cost /= frames
                    cost.backward()
                    #cost2 /= frames
                    cost2.backward()

                    # Updating the parameters
                    self.models[0].optimizer.step()
                    self.models[1].optimizer.step()

                    # Detaching the visible states from GPU for further computation
                    visible_states = visible_states.detach()
                    visible_states2 = visible_states2.detach()

                    # Calculating current's batch MSE
                    batch_mse1 = torch.div(
                        torch.sum(torch.pow(spec_data - visible_states, 2)), batch_size).detach()
                    batch_mse2 = torch.div(
                        torch.sum(torch.pow(x_ - visible_states2, 2)), batch_size).detach()

                    # Calculating the current's batch logarithm pseudo-likelihood
                    batch_pl1 = self.models[0].pseudo_likelihood(spec_data).detach()
                    batch_pl2 = self.models[1].pseudo_likelihood(x_).detach()

                    # Summing up to epochs' MSE and pseudo-likelihood
                    mse2 += (batch_mse1 + batch_mse2)
                    pl2 += (batch_pl1 + batch_pl2)
                    cst2 += (cost.detach() + cost2.detach())

                mse2/=frames
                pl2/=frames
                cst2/=frames

                #cost2 /= frames
                #cost2.backward()
                #self.models[1].optimizer.step()

                mse += mse2
                pl += pl2
                cst += cst2

                if ii % 100 == 99:
                    print('MSE:', (mse/ii).item(), 'Cost:', (cst/ii).item())
                    
                    w8 = self.models[0].W.cpu().detach().numpy()
                    img = _rasterize(w8.T, img_shape=(72, 96), tile_shape=(30, 30), tile_spacing=(1, 1))
                    im = Image.fromarray(img)
                    im.save('w8_spec.png')

                    w8 = self.models[1].W.cpu().detach().numpy()
                    img = _rasterize(w8.T, img_shape=(72, 96), tile_shape=(30, 30), tile_spacing=(1, 1))
                    im = Image.fromarray(img)
                    im.save('w8_gauss.png')

                    x = visible_states[:100].cpu().detach().reshape((100, 6912)).numpy()
                    x = _rasterize(x, img_shape=(72, 96), tile_shape=(10, 10), tile_spacing=(1, 1))
                    im = Image.fromarray(x)
                    im = im.convert("LA")
                    im.save('spectral.png')

                    x = visible_states2[:100].cpu().detach().reshape((100, 6912)).numpy()
                    x = _rasterize(x, img_shape=(72, 96), tile_shape=(10, 10), tile_spacing=(1, 1))
                    im = Image.fromarray(x)
                    im = im.convert("LA")
                    im.save('sample.png')

                inner_trans.update(1)

            mse/=len(batches)
            pl/=len(batches)
            cst/=len(batches)

            logger.info(f'MSE: {mse.item()} | log-PL: {pl.item()} | Cost: {cst.item()}')

            end = time.time()
            self.dump(mse=mse.item(), pl=pl.item(), fe=cst.item(), time=end-start)

        return mse, pl, cst

    def forward(self, x):
        """Performs a forward pass over the data.
        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.
        Returns:
            A tensor containing the DBN's outputs.
       
        """

        # For every possible model        
        x1 = self.models[0].forward(x)
        x2 = self.models[1].forward(x)

        return torch.cat((x1, x2), -1).detach()

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
