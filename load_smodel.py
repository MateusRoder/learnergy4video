from PIL import Image
from learnergy4video.visual.image import _rasterize
import numpy as np
import learnergy4video.visual.tensor as t
import matplotlib.pyplot as plt
import torch, torchvision
import tqdm
import learnergy4video.visual.image as im
from torch.utils.data import DataLoader
from learnergy4video.utils.ucf2 import UCF101
from learnergy4video.utils.hmdb import HMDB51
from learnergy4video.utils.collate import collate_fn

dy = 240
dx = 320
dy = int(.3*dy)
dx = int(.3*dx)
frames_per_clip = 6
batch_size = 50

#dbn, model = torch.load('0fspec_dbn0.pth')[0], torch.load('0fspec_dbn0.pth')[1]
model = torch.load('U1spec_dbm0.pth')
model.cuda()
model.eval()
print(model)

#w8 = model.W.cpu().detach().numpy()
# Creating weights' mosaic
#img = wv.tile_raster_images(model.W.cpu().detach().numpy().T,
#img = wv.tile_raster_images(w8.T, img_shape=(dy, dx), tile_shape=(40, 40), tile_spacing=(1, 1))
#im = Image.fromarray(img)
#im.save('w8_optical_ucf.png')

test = UCF101(train = False, root='/home/roder/RBM-Video/data/UCF-101', annotation_path='/home/roder/RBM-Video/data/ucf_split', frames_per_clip=frames_per_clip, num_workers = 10,
                dim=[dy, dx], chn=1, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=1),
		        torchvision.transforms.PILToTensor()]))
#test = HMDB51(train = False, root='./HMDB51', annotation_path='./HMDB51/splits', frames_per_clip=frames_per_clip, num_workers = 20)

val_batch = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)

inner2 = tqdm.tqdm(total=len(val_batch), desc='Val_Batch', position=2)
rec = torch.zeros((batch_size, 1, dy, dx))
org = torch.zeros((batch_size, 1, dy, dx))
print("Len", rec.size())
i = 0
for x_batch, _ in val_batch:
    val_ = 0
    if model.device == 'cuda':
        x_batch = x_batch.cuda()
    org = x_batch

    for fr in range(1, frames_per_clip):
        x_batch[:, 0, :, :] += x_batch[:, fr, :, :]
        org[:, 0, :, :] += x_batch[:, fr, :, :]

    x_batch = ((x_batch - torch.mean(x_batch, 0, True)) / (torch.std(x_batch, 0, True) + 10e-6)).detach()
    org = ((org - torch.mean(org, 0, True)) / (torch.std(org, 0, True) + 10e-6))

    mse, y = model.reconstruct(x_batch)
    y = y.reshape(y.size(0), dy, dx)
    rec[:, 0,: , :] = y
    i += 1
    break

for b in range(batch_size):
    for i in range(1):
        sample = rec[b, i, :].view(1, dx*dy).cpu().detach().numpy()
        orig = org[b, i, :].view(1, dx*dy).cpu().detach().numpy()

        img = _rasterize(sample, img_shape=(dy, dx),
                                    tile_shape=(1, 1), tile_spacing=(1, 1))

        im = Image.fromarray(img)
        im.save('sample_'+str(i)+'.png')

        img = _rasterize(orig, img_shape=(dy, dx),
                                    tile_shape=(1, 1), tile_spacing=(1, 1))

        im = Image.fromarray(img)
        im.save('orig_'+str(i)+'.png')
    break
