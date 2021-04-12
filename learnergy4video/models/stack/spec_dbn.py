import time
import torch
import tqdm

from torch.utils.data import DataLoader

import learnergy4video.utils.constants as c
import learnergy4video.utils.exception as e
import learnergy4video.utils.logging as l

from learnergy4video.utils.collate import collate_fn
from learnergy4video.core.dataset import Dataset
from learnergy4video.core.model import Model
from learnergy4video.models.binary import RBM
from learnergy4video.models.real import GaussianRBM, SigmoidRBM, EDropoutRBM, EDropoutRBM_Inner, SpecRBM

import os
workers = os.cpu_count()
if workers == None:
    workers = 0
else:
    workers -= 2

logger = l.get_logger(__name__)

MODELS = {
    'bernoulli': RBM,
    'gaussian': GaussianRBM,
    'sigmoid': SigmoidRBM,
    'edrop': EDropoutRBM,
    'edrop_inner': EDropoutRBM_Inner,
    'spectral': SpecRBM
}


class SpecDBN(Model):
    """A SDBN class provides the basic implementation for Deep Belief Networks.
    References:
        G. Hinton, S. Osindero, Y. Teh. A fast learning algorithm for deep belief nets.
        Neural computation (2006).
    """

    def __init__(self, model=['spectral', 'sigmoid'], n_visible=(72, 96), n_hidden=(128,), steps=(1,),
                 learning_rate=(0.1,), momentum=(0,), decay=(0,), temperature=(1,), use_gpu=True):
        """Initialization method.
        Args:
            model (list of str): Indicates which type of RBM should be used to compose the DBN.
            n_visible (tuple): Input shape of visible units.
            n_hidden (tuple): Amount of hidden units per layer.
            steps (tuple): Number of Gibbs' sampling steps per layer.
            learning_rate (tuple): Learning rate per layer.
            momentum (tuple): Momentum parameter per layer.
            decay (tuple): Weight decay used for penalization per layer.
            temperature (tuple): Temperature factor per layer.
            use_gpu (boolean): Whether GPU should be used or not.
        """

        logger.info('Overriding class: Model -> SDBN.')

        # Override its parent class
        super(SpecDBN, self).__init__(use_gpu=use_gpu)

        # Shape of visible input
        self.visible_shape = n_visible

        # Amount of visible units
        self.n_visible = int(n_visible[0]*n_visible[1])

        # Amount of hidden units per layer
        self.n_hidden = n_hidden

        # Number of layers
        self.n_layers = len(n_hidden)

        # Sanity check for the inner models
        if len(model) != self.n_layers:
            for _ in range(len(model)-1, self.n_layers):
                model.append("sigmoid")


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
        self.models = []

        # For every possible layer
        for i in range(self.n_layers):
            # If it is the first layer
            if i == 0:
                # Gathers the number of input units as number of visible units
                n_input = self.n_visible

                # Forcing the Spectral input
                model[i] = 'spectral'

            # If it is not the first layer
            else:
                # Gathers the number of input units as previous number of hidden units
                n_input = self.n_hidden[i-1]

                # After creating the first layer, we need to change the model's type to sigmoid
                #model[i] = 'sigmoid'

            # Creates an RBM
            m = MODELS[model[i]](n_input, self.n_hidden[i], self.steps[i],
                              self.lr[i], self.momentum[i], self.decay[i], self.T[i], use_gpu)

            # Appends the model to the list
            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')
        logger.debug(f'Number of layers: {self.n_layers}.')

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
    def n_layers(self):
        """int: Number of layers.
        """

        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers):
        if not isinstance(n_layers, int):
            raise e.TypeError('`n_layers` should be an integer')
        if n_layers <= 0:
            raise e.ValueError('`n_layers` should be > 0')

        self._n_layers = n_layers

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

    def fit(self, dataset, batch_size=128, epochs=[10], frames=6):
        """Fits a new DBN model.
        Args:
            dataset (torch.utils.data.Dataset | Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (list): Number of training epochs per layer.
        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.
        """

        # Checking if the length of number of epochs' list is correct
        if len(epochs) != self.n_layers:
            # If not, raises an error
            raise e.SizeError(
                f'`epochs` should have size equal as {self.n_layers}')

        # Initializing MSE and pseudo-likelihood as lists
        mse, pl, custo = [], [], []

        # For every possible model (-RBM)
        for i, model in enumerate(self.models):
            logger.info(f'Fitting layer {i+1}/{self.n_layers} ...')

            if i == 0:
                # Fits the RBM
                model_mse, model_pl, cst = model.fit(dataset, batch_size, epochs[i], frames)

                # Appending the metrics
                mse.append(model_mse.item())
                pl.append(model_pl.item())
                custo.append(cst.item())
                self.dump(mse=model_mse.item(), pl=model_pl.item(), fe=cst.item())

            else:
                batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
                for ep in range(epochs[i]): # iterate over epochs for model 'i'
                    logger.info(f'Epoch {ep+1}/{epochs[i]}')
                    mse2, pl2, cst2 = 0, 0, 0

                    inner_trans = tqdm.tqdm(total=len(batches), desc='Batch', position=2)
                    start = time.time()
                    for ii, batch in enumerate(batches):
                        x, y = batch

                        # Checking whether GPU is avaliable and if it should be used
                        if self.device == 'cuda':
                            x = x.cuda()

                        m2, p2, c2 = 0, 0, 0

                        for fr in range(1, frames):
                            x[:, 0, :, :] -= x[:, fr, :, :]

                        x = x[:, 0, :, :]
                        x = x.view(x.size(0), self.models[0].n_visible).detach()
                        x = (x - torch.mean(x, 0, True))/(torch.std(x, 0, True) + 10e-6)
                        
                        for j in range(i): # iterate over trained models
                            x, _ = self.models[j].hidden_sampling(x)
                            #x_ = self.models[j].forward(x_)

                        model_mse, model_pl, ct = model.fit(x, len(x), 1, frames)

                        # Appending the partial metrics
                        m2 += model_mse
                        p2 += model_pl
                        c2 += ct

                        mse2+=m2
                        pl2+=p2
                        cst2+=c2

                        if ii % 100 == 99:
                            print('MSE:', (mse2/ii).item(), 'Cost:', (cst2/ii).item())
                        inner_trans.update(1)

                    mse2/=len(batches)
                    pl2/=len(batches)
                    cst2/=len(batches)
                    mse.append(mse2.item())
                    pl.append(pl2.item())
                    custo.append(cst2.item())

                    logger.info(f'MSE: {mse2.item()} | log-PL: {pl2.item()} | Cost: {cst2.item()}')

                    end = time.time()
                    model.dump(mse=mse2.item(), pl=pl2.item(), fe=cst2.item(), time=end-start)
                    self.dump(mse=mse2.item(), pl=pl2.item(), fe=cst2.item())

        return mse, pl

    def forward(self, samples):
        """Performs a forward pass over the data.
        Args:
            samples (torch.Tensor): An input tensor for computing the forward pass.
        Returns:
            A tensor containing the DBN's outputs.
       
        """

        frames = samples.size(1)

        for fr in range(1, frames):
            samples[:, 0, :, :] -= samples[:, fr, :, :]

        samples = samples[:, 0, :, :]

        # Flattening the samples' batch                    
        samples = samples.view(samples.size(0), self.n_visible)

        # Normalizing the samples' batch
        samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + 10e-6)).detach()


        # For every possible model
        for model in self.models:
            # Calculates the outputs of current model
            samples, _ = model.hidden_sampling(samples)

        return samples.detach()

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
