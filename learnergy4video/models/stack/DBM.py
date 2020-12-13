import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from PIL import Image
from learnergy4video.visual.image import _rasterize

import learnergy4video.utils.exception as e
import learnergy4video.utils.logging as l
from learnergy4video.core import Dataset, Model
from learnergy4video.models.binary import RBM
from learnergy4video.models.real import (GaussianRBM, SigmoidRBM)

logger = l.get_logger(__name__)

MODELS = {
    'bernoulli': RBM,
    'gaussian': GaussianRBM,
    'sigmoid': SigmoidRBM
}


class DBM(Model):
    """A DBM class provides the basic implementation for Deep Boltzmann Machines.
    References:
        .
    """

    def __init__(self, model='gaussian', n_visible=128, n_hidden=(128,), steps=(1,),
                 learning_rate=(0.1,), momentum=(0,), decay=(0,), temperature=(1,), use_gpu=False):
        """Initialization method.
        Args:
            model (str): Indicates which type of RBM should be used to compose the DBM.
            n_visible (int): Amount of visible units.
            n_hidden (tuple): Amount of hidden units per layer.
            steps (tuple): Number of Gibbs' sampling steps per layer.
            learning_rate (tuple): Learning rate per layer.
            momentum (tuple): Momentum parameter per layer.
            decay (tuple): Weight decay used for penalization per layer.
            temperature (tuple): Temperature factor per layer.
            use_gpu (boolean): Whether GPU should be used or not.
        """

        logger.info('Overriding class: Model -> DBM.')

        # Override its parent class
        super(DBM, self).__init__(use_gpu=use_gpu)

        # Amount of visible units
        self.n_visible = n_visible

        # Amount of hidden units per layer
        self.n_hidden = n_hidden

        # Number of layers
        self.n_layers = len(n_hidden)

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Number of Mean-Field steps
        self.m_steps = 25

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

            # If it is not the first layer
            else:
                # Gathers the number of input units as previous number of hidden units
                n_input = self.n_hidden[i-1]

                # After creating the first layer, we need to change the model's type to sigmoid
                model = 'sigmoid'

            # Creates an RBM
            m = MODELS[model](n_input, self.n_hidden[i], self.steps[i],
                              self.lr[i], self.momentum[i], self.decay[i], self.T[i], use_gpu)

            # Appends the model to the list
            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')
        logger.debug('Number of layers: %d.', self.n_layers)

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



    def pretrain(self, dataset, batch_size=128, epochs=[10], frames=6):
        """DBN pre-training phase for further DBM initialization;
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

        def collate_fn(batches):
            try:
                # remove audio from the batch
                batches = [(dd[0], dd[2]) for dd in batches]
                return default_collate(batches)
            except:
                return default_collate(batches)


        # For every possible model (RBM)
        for i, model in enumerate(self.models):
            logger.info(f'Fitting layer {i+1}/{self.n_layers} ...')

            if i == 0:
                # Setting the constraints to pretraining weight
                model.c1 = 2.0
                model.c2 = 1.0

                # Fits the RBM
                model_mse, model_pl, cst = model.fit(dataset, batch_size, epochs[i], frames)

                #model.c1 = 1.0
                #model.c2 = 1.0

                # Appending the metrics
                mse.append(model_mse.item())
                pl.append(model_pl.item())
                custo.append(cst.item())
                self.dump(mse=model_mse.item(), pl=model_pl.item(), fe=cst.item())

            else:
                if i == self.n_layers-1:
                    model.c1 = 1.0
                    model.c2 = 2.0
                else:
                    model.c1 = 2.0
                    model.c2 = 2.0
                batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, collate_fn=collate_fn)
                for ep in range(epochs[i]): # iterate over epochs for model 'i'
                    logger.info(f'Epoch {ep+1}/{epochs[i]}')
                    mse2, pl2, cst2 = 0, 0, 0

                    start = time.time()
                    ii = 1
                    for x, y in tqdm(batches):

                        # Checking whether GPU is avaliable and if it should be used
                        if self.device == 'cuda':
                            x = x.cuda()

                        m2, p2, c2 = 0, 0, 0

                        for fr in range(frames):
                            x_ = x[:, fr, :, :]
                            x_ = x_.view(x.size(0), self.models[0].n_visible).detach()
                            x_ = (x_ - torch.mean(x_, 0, True))/(torch.std(x_, 0, True) + 10e-6)
                        
                            for j in range(i): # iterate over trained models
                                x_ = self.models[j].forward(x_)

                            model_mse, model_pl, ct = model.fit(x_, len(x_), 1, frames)

                            # Appending the partial metrics
                            m2 += model_mse
                            p2 += model_pl
                            c2 += ct
                        m2/=frames
                        p2/=frames
                        c2/=frames
                        mse2+=m2
                        pl2+=p2
                        cst2+=c2
                        if ii % 100 == 99:
                            print('MSE:', (mse2/ii).item(), 'Cost:', (cst2/ii).item())
                        #inner_trans.update(1)
                        ii += 1

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

            # Gathers the transform callable from current dataset
            transform = None

            #if i < (len(self.models)-1) :
            #    with torch.no_grad(): model.W /= 2

        for model in self.models:
            model.c1 = 1.0
            model.c2 = 1.0

        return mse, pl

    def mean_field(self, samples):
        """Mean-Field approximation.
        Args:
            samples (torch.tensor): Samples containing the training data.
        Returns:
            Approximated probabilities, i.e., P(v|h1), P(h1|v,h2), etc.
        """

        # Useful variables initialization
        hidden_probs = samples
        samples2 = samples
        mf = [samples2]

        # Performing the fast inference
        for model in self.models:
            # Flattening the hidden probabilities
            hidden_probs = hidden_probs.reshape(samples.size(0), model.n_visible)

            # Performing a hidden layer sampling
            hidden_probs, _ = model.hidden_sampling(hidden_probs)
            mf.append(hidden_probs.detach())

        # Performing the variational inference:
        for i in range(self.m_steps):
            for j in range(self.n_layers-1):
                mf[j] = torch.sigmoid(
                    	torch.matmul(samples2, self.models[j].W) + torch.matmul(mf[j + 1], self.models[j + 1].W.t()) +
                    	self.models[j].b).detach()
                samples2 = mf[j]

            mf[j + 1] = torch.sigmoid(torch.matmul(mf[j], self.models[j+1].W) + self.models[j+1].b).detach()
            samples2 = samples

        mf[0] = torch.sigmoid(
                torch.matmul(mf[1], self.models[0].W.t()) + self.models[0].a).detach()

        return mf

    def gibbs_sampling(self, samples):
        """Performs the whole Gibbs sampling procedure.

        Args:
            samples (torch.Tensor): A tensor incoming from the visible layer (image samples).

        Returns:
            cdk (list): The states of all the hidden layers.

        """

        # Performing the Contrastive Divergence
        hidden_probs = samples
        cdk = []

        # For every possible model (RBM)
        for model in self.models:
            # Calculating positive phase hidden probabilities and states
            pos_hidden_probs, pos_hidden_states = model.hidden_sampling(hidden_probs)

            # Initially defining the negative phase
            neg_hidden_states = pos_hidden_states

            # Performing the Contrastive Divergence
            for _ in range(self.steps[0]):
                # Calculating visible probabilities and states
                _, visible_states = model.visible_sampling(
                    neg_hidden_states, True)

                # Calculating hidden probabilities and states
                neg_hidden_probs, neg_hidden_states = model.hidden_sampling(
                    visible_states, True)

            hidden_probs = neg_hidden_probs.detach()
            cdk.append(visible_states.detach())

        return cdk
                
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
            ii = 1
            #cst = torch.zeros((self.n_layers), dtype=torch.float, device=self.device)

            # For every batch
            for samples, y in tqdm(batches):
                #cost = 0
                cst2 = 0
                #cost = torch.zeros((self.n_layers), dtype=torch.float, device=self.device)
                #cst2 = torch.zeros((self.n_layers), dtype=torch.float, device=self.device)
                if self.device == 'cuda':
                    samples = samples.cuda()

                mse2, pl2 = 0, 0

                for fr in range(frames):
                    torch.autograd.set_detect_anomaly(True)
                    # Flattening the samples' batch
                    sps = samples[:, fr, :, :]
                    sps = sps.view(sps.size(0), self.n_visible)

                    # Normalizing the samples' batch
                    sps = ((sps - torch.mean(sps, 0, True)) / (torch.std(sps, 0, True) + 10e-6)).detach()
                
                    # Checking whether GPU is avaliable and if it should be used
                    if self.device == 'cuda':
                    # Applies the GPU usage to the data
                        sps = sps.cuda()

                    # Performs the mean-field approximation
                    mf = self.mean_field(sps)

                    # Performs the Gibbs sampling procedure
                    visible_states = self.gibbs_sampling(sps)

                    # Calculates the loss for further gradients' computation
                    for idx, model in enumerate(self.models):
                        # Initializing the gradient
                        model.optimizer.zero_grad()

                        cost = torch.mean(model.energy(mf[idx])) - \
                                  torch.mean(model.energy(visible_states[idx]))
                        cst2+=cost.detach().item()
                        #cost[idx] = cost[idx] + torch.mean(model.energy(mf[idx])) - \

                        # Computing the gradients    
                        cost.backward()

                        # Updating the parameters
                        model.optimizer.step()

                    # Detaching the visible states from GPU for further computation
                    #visible_states = visible_states.detach()

                    # Gathering the size of the batch
                    batch_size2 = sps.size(0)

                    # Calculating current's batch MSE
                    batch_mse = torch.div(
                        torch.sum(torch.pow(sps - visible_states[0], 2)), batch_size2).detach()

                    # Calculating the current's batch logarithm pseudo-likelihood
                    #batch_pl = self.pseudo_likelihood(sps).detach()

                    # Updating the parameters
                    #for model in self.models:
                        #model.optimizer.step()

                    # Summing up to epochs' MSE and pseudo-likelihood
                    mse2 += batch_mse
                    #pl2 += batch_pl
                    #cst2 += cost.detach()

                    # Initializing the gradient
                    #for model in self.models:
                        #model.optimizer.zero_grad()


                mse2 /= frames
                cst2 /= (frames*self.n_layers)
                #pl2 /= frames
                #cst2 /= frames

                mse += mse2
                #pl  += pl2
                cst += cst2
                if ii % 100 == 99:
                    print('MSE:', (mse/ii).item(), 'Cost:', cst/ii)
                    w8 = self.models[0].W.cpu().detach().numpy()
                    #img = _rasterize(w8.T, img_shape=(96, 128), tile_shape=(30, 30), tile_spacing=(1, 1))
                    img = _rasterize(w8.T, img_shape=(72, 96), tile_shape=(30, 30), tile_spacing=(1, 1))
                    im = Image.fromarray(img)
                    im.save('w8_fit_ucf101.png')

                ii += 1

            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            #pl /= len(batches)
            #cst /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            #self.dump(mse=mse.item(), pl=pl.item(), fe=cst.item(), time=end-start)
            self.dump(mse=mse.item(), time=end-start)
            
            #logger.info(f'MSE: {mse} | log-PL: {pl} | Cost: {cst}')
            logger.info(f'MSE: {mse}')

        return mse #,pl, cst


    def reconstruct(self, dataset):
        """Reconstructs batches of new samples.
        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).
        """

        logger.info('Reconstructing new samples ...')

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        # For every batch
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(batch_size, self.models[0].n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Applying the initial hidden probabilities as the samples
            hidden_probs = samples

            # For every possible model (RBM)
            for model in self.models:
                # Flattening the hidden probabilities
                hidden_probs = hidden_probs.reshape(
                    batch_size, model.n_visible)

                # Performing a hidden layer sampling
                hidden_probs, _ = model.hidden_sampling(
                    hidden_probs)

            # Applying the initial visible probabilities as the hidden probabilities
            visible_probs = hidden_probs

            # For every possible model (RBM)
            for model in reversed(self.models):
                # Flattening the visible probabilities
                visible_probs = visible_probs.reshape(
                    batch_size, model.n_hidden)

                # Performing a visible layer sampling
                visible_probs, visible_states = model.visible_sampling(
                    visible_probs)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

            # Summing up to reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info('MSE: %f', mse)

        return mse, visible_probs

    def forward(self, x):
        """Performs a forward pass over the data.
        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.
        Returns:
            A tensor containing the DBN's outputs.
        """

        # For every possible model
        #for model in self.models:
            # Calculates the outputs of current model
            #x, _ = model.hidden_sampling(x)

        return self.mean_field(x)[self.n_layers-1].detach()
