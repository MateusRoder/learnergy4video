import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

#from learnergy4video.models.real import GaussianConvRBM as CRBM
from learnergy4video.models.stack import CDBN, SpecCDBN
#from learnergy4video.models.stack.dbn import DBN
from learnergy4video.utils.ucf2 import UCF101
from learnergy4video.utils.hmdb import HMDB51

import os
workers = os.cpu_count()
if workers == None:
    workers = 0
else:
    workers -= 2


if __name__ == '__main__':

    dy = 240
    dx = 320
    dy = int(.4*dy)
    dx = int(.4*dx)
    frames_per_clip = 6

    # Defining some input variables
    n_layers = 2
    n_filters = [16*2, 16*2, 16*2]
    f_shape = [(7, 7), (5, 5), (5, 5)]
    n_channels = 1
    batch_size = 2**7
    n_classes = 5
    fine_tune_epochs = 3

    for j in range(1):
        np.random.seed(j)
        torch.manual_seed(j)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #name = 'H1spec_cdbn'+str(j)+'.pth'
        #name = 'H1cdbn_std'+str(j)+'.pth'
        name = 'U1spec_cdbn'+str(j)+'.pth'
        #name = 'U1cdbn_std'+str(j)+'.pth'
        train = UCF101(root='/home/roder/RBM-Video/UCF-101', annotation_path='/home/roder/RBM-Video/ucf_split', 
                        frames_per_clip=frames_per_clip, num_workers=workers,
        #train = HMDB51(train = True, root='./HMDB51', annotation_path='./HMDB51/splits', frames_per_clip=frames_per_clip, num_workers = 20,
			dim=[dy, dx], chn=n_channels, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=n_channels),
		        torchvision.transforms.PILToTensor()]))


        # Creating the model
        model = SpecCDBN(visible_shape=(dy, dx), filter_shape=f_shape,
             n_filters=n_filters, n_channels=n_channels, n_layers=n_layers,
             learning_rate=(0.000001, 0.000001, 0.0001), momentum=(0, 0, 0), decay=(0, 0, 0),
             #learning_rate=(0.000001, 0.000001, 0.000001), momentum=(0.9, 0.9, 0.9), decay=(0.00001, 0.00001, 0.00001),
             use_gpu=True)
     
            
        # Training the model
        model.fit(train, batch_size=batch_size, epochs=(3, 3, 3)) # convergence stucks in 5 epochs!
        torch.save(model, name)

        import classification_sdbn
        from classification_sdbn import exec_class
        #import classification_dbn
        #from classification_dbn import exec_class
        exec_class(name, j)
