import torch
import numpy as np

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
        
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift(X):
    # batch*channel*...*2
    if X.dim() < 4:
        X = torch.cat((X[:,:,:,None], X[:,:,:,None]), -1)
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    for dim in range(1, len(real.size())): # 1 channel
        real = roll_n(real, axis=dim, n=int(np.ceil(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim, n=int(np.ceil(imag.size(dim) / 2)))
        
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    
    return torch.cat((real, imag),dim=-1)

def ifftshift(X):
    # batch*channel*...*2
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    #for dim in range(len(real.size()) - 1, 1, -1):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=int(np.floor(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim, n=int(np.floor(imag.size(dim) / 2)))
        
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)

    return torch.cat((real, imag), dim=-1)
