import torch
import torchvision
import learnergy4video.visual.image as im

# Creating testing dataset
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Loading pre-trained model
model = torch.load('dbm_model.pth')

# Reconstructing test set
#rec_mse, v = model.reconstruct(test)
im.create_mosaic(model.models[0].W)