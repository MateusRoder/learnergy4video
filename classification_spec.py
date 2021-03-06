import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from numpy import savetxt
from torch.utils.data import DataLoader, Dataset

from learnergy4video.utils.collate import collate_fn
from learnergy4video.utils.hmdb import HMDB51
from learnergy4video.utils.ucf2 import UCF101

def exec_class(name, seed):
    # Defining some input variables
    batch_size = 2**7
    dv = int(0.15*batch_size)
    print("Train samples:", batch_size-dv, "Validation:", dv)
    n_classes = 5
    fine_tune_epochs = 3
    frames_per_clip = 6

    model = torch.load(name)
    dy = model.visible_shape[0]
    dx = model.visible_shape[1]

    print(model, "> Hidden Neurons:", model.n_hidden)
    tam = len(model.n_hidden)
    nhidden = model.n_hidden[tam-1]
    
    # Creating the Fully Connected layer to append on top of an RBM
    fc = torch.nn.Linear(nhidden, nhidden//2)
    fc2 = torch.nn.Linear(nhidden//2, n_classes)

    # Check if model uses GPU
    if model.device == 'cuda':
        fc.cuda()
        fc2.cuda()
        model.cuda()

    model.eval()
    print("Model Loaded!")

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [optim.Adam(fc.parameters(), lr=1e-3),
                 optim.Adam(fc2.parameters(), lr=1e-3)]
    for i in range(1, tam):
        optimizer.append(optim.Adam(model.models[i].parameters(), lr=1e-5))

    # Creating training and validation batches
    train = HMDB51(train = True, root='/home/roder/RBM-Video/HMDB51', annotation_path='/home/roder/RBM-Video/HMDB51/splits', 
                    frames_per_clip=frames_per_clip, num_workers = 10,
                    dim=[dy, dx], chn=1, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=1),
		        torchvision.transforms.PILToTensor()]))
    #train = HMDB51(train = True, root='./HMDB51', annotation_path='./HMDB51/splits', frames_per_clip=frames_per_clip, num_workers = 20)


    train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)
    #val_batch = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # For amount of fine-tuning epochs
    ii=0
    save_loss = []
    save_loss2 = []
    for e in range(fine_tune_epochs):
        print(f'\nEpoch {e+1}/{fine_tune_epochs}')

        # Resetting metrics
        train_loss, val_acc, test_loss, ac, ii = 0, 0, 0, 0, 0

        # For every possible batch
        inner = tqdm.tqdm(total=len(train_batch), position=1)
        for x_batch, y_batch in train_batch:

            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            tr_loss, ts_loss = 0, 0
            acc, vacc = 0, 0
            y_t = y_batch[:dv]
            y_batch = y_batch[dv:]
            
            x_t = x_batch[:dv, :]
            x_ = x_batch[dv:, :]

            # Passing the batch down the model
            y = model(x_)
            yt = model(x_t)

            for fr in range(frames_per_clip):

                for opt in optimizer:
                    opt.zero_grad()

                # Calculating the fully-connected outputs
                y2 = fc(y[:, fr, :])
                y2 = fc2(y2)
                yt2 = fc(yt[:, fr, :])
                yt2 = fc2(yt2)
                _, preds = torch.max(y2, 1)
                _, preds2 = torch.max(yt2, 1)

                acc += torch.mean((torch.sum(preds == y_batch).float()) / preds.size(0)).detach().item()
                vacc += torch.mean((torch.sum(preds2 == y_t).float()) / preds2.size(0)).detach().item()

                # Calculating loss
                loss = criterion(y2, y_batch)
                loss2 = criterion(yt2, y_t)

                # Propagating the loss to calculate the gradients
                loss.backward()

                # For every possible optimizer
                for opt in optimizer:
                    # Performs the gradient update
                    opt.step()

                # Adding current batch loss
                tr_loss += loss.detach().item()
                ts_loss += loss2.detach().item()

            tr_loss/=frames_per_clip
            ts_loss/=frames_per_clip
            acc/=frames_per_clip
            vacc/=frames_per_clip
            train_loss += tr_loss
            test_loss += ts_loss
            ac += acc
            val_acc += vacc
            ii += 1
            inner.set_description('Loss: %g x %g| Acc: %g | V_acc: %g' % (train_loss/ii, test_loss/ii, ac/ii, val_acc/ii))
            inner.update(1)
        ac/=len(train_batch)
        test_loss/=len(train_batch)
        #print(f'\nMean Loss: {train_loss/len(train_batch)} | Acc: {ac}')
        #group_model = [model, fc]
        group_model = [model, fc, fc2]
        torch.save(group_model, 'tuned_'+str(name))
        save_loss.append((train_loss/len(train_batch)))
        save_loss2.append((test_loss))

    #group_model = [model, fc]
    group_model = [model, fc, fc2]
    torch.save(group_model, 'tuned_'+str(name))
    savetxt(name[:-8]+'train_loss'+str(seed)+'.csv', save_loss, delimiter=',')
    savetxt(name[:-8]+'val_loss'+str(seed)+'.csv', save_loss2, delimiter=',')

    ## validation ##
    frames_per_clip = 6
    batch_size = 2**7
    test = HMDB51(train = False, root='/home/roder/RBM-Video/HMDB51', annotation_path='/home/roder/RBM-Video/HMDB51/splits', 
                    frames_per_clip=frames_per_clip, num_workers = 10,
                    dim=[dy, dx], chn=1, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=1),
		        torchvision.transforms.PILToTensor()]))
    #test = HMDB51(train = False, root='./HMDB51', annotation_path='./HMDB51/splits', frames_per_clip=frames_per_clip, num_workers = 20)
    val_batch = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn)
    print("\n")
    val_acc = []
    inner2 = tqdm.tqdm(total=len(val_batch), position=2)
    for x_batch, y_batch in val_batch:
        val_ = 0
        if model.device == 'cuda':
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            y = model(x_batch)
            for fr in range(frames_per_clip):
                # Calculating the fully-connected outputs
                y2 = fc(y[:, fr, :])
                y2= fc2(y2)
                _, preds = torch.max(y2, 1)
                val_ += torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0)).detach().item()
            val_ /= frames_per_clip

        val_acc.append(val_)
        inner2.set_description('Acc: %g' % np.mean(val_acc))
        inner2.update(1)

    savetxt(name[:-3]+'txt', np.array(val_acc))
    print(f'\nVal Accuracy: {np.mean(val_acc)}')

#exec_class('H1spec_rbm0.pth', 0)
