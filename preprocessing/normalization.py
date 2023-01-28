import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
 
def normalize_input_train(X_train, reshaped = True, norm = "standard", save_files = False, norm_path = "../training/norm_files/", save_path = "../training/training_data/"):
    if reshaped:
        train_mu = np.mean(X_train, axis = 1)[:, np.newaxis]
        train_std = np.std(X_train, axis = 1)[:, np.newaxis]
        train_min = X_train.min(axis = 1)[:, np.newaxis]
        train_max = X_train.max(axis = 1)[:, np.newaxis]
    
    else:
        train_mu = np.mean(X_train, axis = (1,2,3))[:, np.newaxis]
        train_std = np.std(X_train, axis = (1,2,3))[:, np.newaxis]
        train_min = X_train.min(axis = (1,2,3))[:, np.newaxis]
        train_max = X_train.max(axis = (1,2,3))[:, np.newaxis]
        
    if norm == "standard":
        inp_sub = train_mu
        inp_div = train_std
        
    elif norm == "range":
        inp_sub = train_min
        inp_div = train_max - train_min
        
    #normalizing
    X_train = ((X_train - inp_sub)/inp_div).transpose()
    #normalized
    
    print("X_train shape: ")
    print(X_train.shape)
    print("INP_SUB shape: ")
    print(inp_sub.shape)
    print("INP_DIV shape: ")
    print(inp_div.shape)
    
    if save_files:
        with open(save_path + "train_input.npy", 'wb') as f:
            np.save(f, np.float32(X_train))
        np.savetxt(norm_path + "inp_sub.txt", inp_sub, delimiter=',')
        np.savetxt(norm_path + "inp_div.txt", inp_div, delimiter=',')
    
    return X_train, inp_sub, inp_div


def normalize_input_val(X_val, inp_sub, inp_div, save_files = False, save_path = "../training/training_data/"):
    #normalizing
    X_val = ((X_val - inp_sub)/inp_div).transpose()
    print("X_val shape: ")
    print(X_val.shape)
    print("INP_SUB shape: ")
    print(inp_sub.shape)
    print("INP_DIV shape: ")
    print(inp_div.shape)
    
    if save_files:
        with open(save_path + "val_input.npy", 'wb') as f:
            np.save(f, np.float32(X_val))
    
    return X_val


def normalize_target_train(y_train, reshaped = True, save_files = False, save_path = "../training/training_data/"):
    
    # specific heat of air = 1004 J/ K / kg
    # latent heat of vaporization 2.5*10^6

    heatScale = 1004
    moistScale = 2.5e6
    outscale = np.concatenate((np.repeat(heatScale, 30), np.repeat(moistScale, 30)))
    
    if reshaped:
        y_train[0:30,:] = y_train[0:30,:]*outscale[0:30, np.newaxis]
        y_train[30:60,:] = y_train[30:60,:]*outscale[30:60, np.newaxis]
    else:
        y_train[0:30,:] = y_train[0:30,:]*outscale[0:30, np.newaxis, np.newaxis, np.newaxis]
        y_train[30:60,:] = y_train[30:60,:]*outscale[30:60, np.newaxis, np.newaxis, np.newaxis]        
    
    y_train = y_train.transpose()
    print("y shape: ")
    print(y_train.shape)
    print("outscale shape: ")
    print(outscale.shape)
    
    if save_files:
        with open(save_path + "train_target.npy", 'wb') as f:
            np.save(f, np.float32(y_train))

    return y_train


def normalize_target_val(y_val, reshaped = True, save_files = False, save_path = "../training/training_data/"):
    
    # specific heat of air = 1004 J/ K / kg
    # latent heat of vaporization 2.5*10^6

    heatScale = 1004
    moistScale = 2.5e6
    outscale = np.concatenate((np.repeat(heatScale, 30), np.repeat(moistScale, 30)))
    
    if reshaped:
        y_val[0:30,:] = y_val[0:30,:]*outscale[0:30, np.newaxis]
        y_val[30:60,:] = y_val[30:60,:]*outscale[30:60, np.newaxis]
    else:
        y_val[0:30,:] = y_val[0:30,:]*outscale[0:30, np.newaxis, np.newaxis, np.newaxis]
        y_val[30:60,:] = y_val[30:60,:]*outscale[30:60, np.newaxis, np.newaxis, np.newaxis]        
    
    y_val = y_val.transpose()
    print("y shape: ")
    print(y_val.shape)
    print("outscale shape: ")
    print(outscale.shape)
    
    if save_files:
        with open(save_path + "val_target.npy", 'wb') as f:
            np.save(f, np.float32(y_val))
            
    return y_val