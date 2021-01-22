import numpy as np
import torch
import torchvision

from learnergy4video.models.stack.dbn import DBN
from learnergy4video.models.stack.spec_dbn import SpecDBN
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
    dy = int(.3*dy)
    dx = int(.3*dx)
    frames_per_clip = 6

    for j in range(6):
        np.random.seed(j)
        torch.manual_seed(j)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        name = 'H1spec_rbm'+str(j)+'.pth'

        #train = UCF101(root='./UCF-101', annotation_path='./ucf_split',
        train = HMDB51(root='./HMDB51', annotation_path='./HMDB51/splits',
                        frames_per_clip=frames_per_clip, num_workers=workers,
                        dim=[dy, dx], chn=1, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=1),               
		        torchvision.transforms.PILToTensor()]))


        batch_size = 2**7
        ep = 3
        hidden = 2000
     
        #model = DBN(model=['gaussian'], n_visible=(dy, dx), n_hidden=(hidden,), steps=(1,),
        model = SpecDBN(model=['spectral'], n_visible=(dy, dx), n_hidden=(hidden,), steps=(1,),
                    learning_rate=(0.0001,), momentum=(0.5,), decay=(0,), temperature=(1,),
                    use_gpu=True)


        # Training a Spec-RBM model (1 hidden layer stands for an RBM)
        mse, pl = model.fit(train, batch_size=batch_size, epochs=[ep], frames=frames_per_clip)
        torch.save(model, name)

        # Training the classifier Spec-model
        import classification_spec
        from classification_spec import exec_class

        # Training the classifier standard-model
        #import classification_std
        #from classification_std import exec_class

        exec_class(name, j)
