import numpy as np
import torch
import torchvision

from learnergy4video.models.stack.DBM import DBM
from learnergy4video.utils.ucf2 import UCF101
#from learnergy4video.utils.hmdb import HMDB51

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

    for j in range(1):
        np.random.seed(j)
        torch.manual_seed(j)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        name = 'U1dbm_std'+str(j)+'.pth'
        train = UCF101(root='/home/roder/RBM-Video/UCF-101', annotation_path='/home/roder/RBM-Video/ucf_split',
                        frames_per_clip=frames_per_clip, num_workers=workers,
                        dim=[dy, dx], chn=1, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=1),               
		        torchvision.transforms.PILToTensor()]))
        #train = HMDB51(train = True, root='./HMDB51', annotation_path='./HMDB51/splits', frames_per_clip=frames_per_clip, num_workers = 20)

        batch_size = 2**7
        ep = 3
        hidden = 2000
     
        model = DBM(model='gaussian', n_visible=dx*dy, n_hidden=(hidden, hidden), steps=(1,1),
                    learning_rate=(0.0001, 0.00005), momentum=(0.5, 0.5), decay=(0,0), temperature=(1,1),
                    use_gpu=True)

        #model = torch.load("U1dbm_std0 (cópia).pth")
        #model.eval()
        #model.cuda()
        # Training a DBM
        mse, pl = model.pretrain(train, batch_size=batch_size, epochs=[ep, ep], frames=frames_per_clip)
        torch.save(model, name)

        mse = model.fit(train, batch_size=batch_size, epochs=ep)
        torch.save(model, name)

        #import classification_dbm
        #from classification_dbm import exec_class
        #exec_class(name, j)
