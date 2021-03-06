import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from numpy import savetxt
from torch.utils.data import DataLoader, Dataset

from learnergy4video.utils.collate import collate_fn
from learnergy4video.utils.ucf2 import UCF101
from learnergy4video.utils.hmdb import HMDB51


def exec_class(name, seed):
    torch.manual_seed(seed)
    # Defining some input variables
    batch_size = 2**7
    channel = 1
    dv = int(0.15*batch_size)
    print("Train samples:", batch_size-dv, "Validation:", dv)
    n_classes = 5
    fine_tune_epochs = 3
    frames_per_clip = 6

    model = torch.load(name)

    dy = model.visible_shape[0]
    dx = model.visible_shape[1]

    try: 
        tam = len(model.models)
    except:
        tam = 1

    # Creating the Fully Connected layer to append on top of CDBN
    try:
        in_shape = (model.hidden_shape[0]//2+1) * (model.hidden_shape[1]//2+1) * model.n_filters[tam-1]
        #in_shape = model.hidden_shape[0] * model.hidden_shape[1] * model.n_filters[tam-1] # without max-pooling
    except:
        in_shape = model.hidden_shape[0] * model.hidden_shape[1] * model.n_filters
    fc = nn.Linear(in_shape, n_classes)
    #fc2 = nn.Linear(in_shape//2, n_classes)

    print("FC Input:", in_shape)

    # Check if model uses GPU
    if model.device == 'cuda':
        # If yes, put fully-connected on GPU
        fc.cuda()
        #fc2.cuda()
        model.cuda()

    model.eval()
    print("Model Loaded!")

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [optim.Adam(fc.parameters(), lr=1e-3)]
                 #optim.Adam(fc2.parameters(), lr=1e-3)]
    try:
        for i in range(1, tam):
            optimizer.append(optim.Adam(model.models[i].parameters(), lr=1e-5)) #Estudar 10e-7
    except:
            optimizer.append(optim.Adam(model.parameters(), lr=1e-4)) #Estudar 10e-7

    # Creating training and validation batches
    train = UCF101(train = True, root='./full_data/UCF-101', annotation_path='./full_data/ucf_split', frames_per_clip=frames_per_clip, num_workers = 20,
    #train = HMDB51(train = True, root='./HMDB51', annotation_path='./HMDB51/splits', frames_per_clip=frames_per_clip, num_workers = 20,
			dim=[dy, dx], chn=1, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
		        #torchvision.transforms.RandomCrop((dy, dx)),
		        #torchvision.transforms.RandomRotation(10),
		        #torchvision.transforms.RandomHorizontalFlip(p=0.60),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=channel),
		        torchvision.transforms.PILToTensor()]))


    train_batch = DataLoader(train, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=20, collate_fn=collate_fn)
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
            for fr in range(frames_per_clip):
                for opt in optimizer:
                    opt.zero_grad()

                # Flatenning the samples batch
                x_ = x_batch[:, fr, :, :]/x_batch.max()
                x_ = x_.reshape((len(x_), channel, dy, dx))
                x_t = x_[:dv]
                x_ = x_[dv:]

        
                # Passing the batch down the model
                y = model(x_).detach()
                yt = model(x_t).detach()

                # Calculating the fully-connected outputs
                y = y.reshape(y_batch.size(0), in_shape)
                yt = yt.reshape(y_t.size(0), in_shape)
                y = fc(y)#.detach()
                #y = fc2(y)
                yt = fc(yt)#.detach()
                #yt = fc2(yt)
                _, preds = torch.max(y, 1)
                _, preds2 = torch.max(yt, 1)

                # Calculating loss
                loss = criterion(y, y_batch)        
                loss2 = criterion(yt, y_t)        

                # Propagating the loss to calculate the gradients
                loss.backward()

                # For every possible optimizer
                for opt in optimizer:
                    # Performs the gradient update
                    opt.step()

                # Adding current batch loss
                tr_loss += loss.detach().item()
                ts_loss += loss2.detach().item()

                acc += torch.mean((torch.sum(preds == y_batch).float()) / preds.size(0)).detach().item()
                vacc += torch.mean((torch.sum(preds2 == y_t).float()) / preds2.size(0)).detach().item()
                
            tr_loss /= frames_per_clip
            ts_loss /= frames_per_clip
            acc /= frames_per_clip
            vacc /= frames_per_clip
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
        #group_model = [model, fc, fc2]
        group_model = [model, fc]
        torch.save(group_model, 'tuned_'+str(name))
        save_loss.append((train_loss/len(train_batch)))
        save_loss2.append((test_loss))

    group_model = [model, fc]
    #group_model = [model, fc, fc2]
    torch.save(group_model, 'tuned_'+str(name))
    savetxt(name[:-8]+'train_loss'+str(seed)+'.csv', save_loss, delimiter=',')
    savetxt(name[:-8]+'val_loss'+str(seed)+'.csv', save_loss2, delimiter=',')

    ## validation ##
    frames_per_clip = 6
    batch_size = 2**7
    test = UCF101(train = False, root='./full_data/UCF-101', annotation_path='./full_data/ucf_split', frames_per_clip=frames_per_clip, num_workers = 20,
    #test = HMDB51(train = False, root='./HMDB51', annotation_path='./HMDB51/splits', frames_per_clip=frames_per_clip, num_workers = 20,
			dim=[dy, dx], chn=1, transform=torchvision.transforms.Compose([
		        torchvision.transforms.ToPILImage(),
		        #torchvision.transforms.RandomCrop((dy, dx)),
		        #torchvision.transforms.RandomRotation(10),
		        #torchvision.transforms.RandomHorizontalFlip(p=0.60),
        		torchvision.transforms.Resize((dy, dx)),
		        torchvision.transforms.Grayscale(num_output_channels=channel),
		        torchvision.transforms.PILToTensor()]))
    val_batch = DataLoader(test, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=20, collate_fn=collate_fn)
    print("\n")
    val_acc = []
    inner2 = tqdm.tqdm(total=len(val_batch), position=2)
    for x_batch, y_batch in val_batch:
        val_ = 0
        if model.device == 'cuda':
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        for fr in range(frames_per_clip):
            x_ = x_batch[:, fr, :, :]/x_batch.max()
            x_ = x_.reshape((len(x_), channel, dy, dx))

            y = model(x_)
            y = y.reshape(y_batch.size(0), in_shape)
            y = fc(y)
            #y = fc2(y)
            _, preds = torch.max(y, 1)
            val_ += torch.mean((torch.sum(preds == y_batch).float()) / x_.size(0)).detach().item()

        val_ /= frames_per_clip
        #print(f'Batch Accuracy: {val_}')
        val_acc.append(val_)
        inner2.set_description('Acc: %g' % np.mean(val_acc))
        inner2.update(1)
    #val_acc /= len(val_batch)
    #val_acc = np.sum(val_acc)/len(val_acc)
    savetxt(name[:-3]+'txt', np.array(val_acc))
    print(f'\nVal Accuracy: {np.mean(val_acc)}')

#exec_class('cdbn_model.pth', 0)
#exec_class('U1cdbn_std0.pth', 0)
