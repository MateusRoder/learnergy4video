import torch
import torchvision

from learnergy4video.models.stack import DBN
import learnergy4video.visual.image as im
import learnergy4video.visual.tensor as t

# Creating training and testing dataset
train = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Creating a DBN
model = DBN(model='sigmoid', n_visible=784, n_hidden=(625, 1089), steps=(1, 1),
            learning_rate=(0.01, 0.01), momentum=(0, 0), decay=(0, 0), temperature=(1, 1),
            use_gpu=True)

# Training a DBN
model.fit(train, batch_size=100, epochs=(5, 5))

# Reconstructing test set
rec_mse, v = model.reconstruct(test)
t.show_tensor(v[1].reshape(28, 28))
t.show_tensor(test.data[1].reshape(28, 28))
im.create_mosaic(model.models[0].W.detach().cpu())

# Saving model
#torch.save(model, 'model.pth')

# Checking the model's history
#for m in model.models:
#    print(m.history)
