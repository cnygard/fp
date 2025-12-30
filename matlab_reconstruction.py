import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from util import *

debug = False

def reconstruction(observations, Theta, num_epochs, config):
    num_samples = observations.shape[0]
    num_angles = Theta.shape[0]
    
    print(f"Number of observations: {num_samples}, Number of angles: {num_angles}")
    # guess = np.zeros((observations.shape[1], observations.shape[2]))
    # guess = np.copy(observations[0,:,:])

    guess = np.sum(observations, axis=0)
    guess = cv2.resize(guess,
                       (guess.shape[0] * config['upsample'],
                        guess.shape[1] * config['upsample']))
    guessFT = F(guess)

    # debug view
    # region = np.copy(guessFT)
    region = np.zeros(guessFT.shape)

    kx,ky,dkx,dky = kspace_parameterize(Theta, config)

    # need angle per measurement to reconstruct
    # call replace-angle on sample
    prog_bar = tqdm(range(num_epochs))
    first = True
    for _ in prog_bar:
        for i in range(num_samples):
            # image = sample # reproject current estimate to same angle as this observation
            lx,ly,hx,hy = find_bounds(kx[i].item(),ky[i].item(),dkx,dky,config)
            r = guessFT[ly:hy, lx:hx]
            r_masked = r * config['M'].detach().cpu().numpy()
            low_res_guess = iF(r_masked)
            measurement = config['upsample']**2 * observations[i,:,:]
            low_res_guess = measurement * np.exp(1j*np.angle(low_res_guess)) # replace amplitude but preserve phase
            low_res_guess_FT = F(low_res_guess)
            guessFT[ly:hy,lx:hx] =     \
                guessFT[ly:hy,lx:hx] + \
                config['M'].detach().cpu().numpy() *     \
                (low_res_guess_FT - r_masked)

            if debug and first:
                toBeChanged = region[ly:hy,lx:hx]
                toBeAdded = config['M'].detach().cpu().numpy() * \
                            (low_res_guess_FT - r_masked)
                region[ly:hy,lx:hx] =     \
                    region[ly:hy,lx:hx] + \
                    config['M'].detach().cpu().numpy() *    \
                    (low_res_guess_FT - r_masked)
                guess = iF(guessFT)
                img1 = 'Measurement', observations[i,:,:]
                img2 = 'Guess', np.real(guess)
                img3 = 'Guess Spectrum', np.log(np.real(region))
                debug_print(img1, img2, img3)
        first = False
    return iF(guessFT)

def sim(gt_sample, Theta, num_epochs, config):
    num_samples = Theta.shape[0]
    observations = np.empty((num_samples,
                             gt_sample.shape[0],
                             gt_sample.shape[1]))
    for i in range(num_samples):
        observations[i,:,:] = np.copy(gt_sample)
    return reconstruction(observations, Theta, num_epochs, config)

def real(observations, Theta, num_epochs, config):
    return reconstruction(observations, Theta, num_epochs, config)

def find_bounds(x,y,dx,dy,config):
    center_of_output = (config['out_N']+1) / 2
    centerX = matlab_round(center_of_output - x / dx)
    centerY = matlab_round(center_of_output - y / dy)
    lowX  = matlab_round(centerX - (config['N']-1)/2)
    lowY  = matlab_round(centerY - (config['N']-1)/2)
    highX = matlab_round(centerX + (config['N']-1)/2) + 1
    highY = matlab_round(centerY + (config['N']-1)/2) + 1
    return lowX, lowY, highX, highY

def kspace_parameterize(Theta, config):
    kx = config['k'] * Theta[:,0]
    ky = config['k'] * Theta[:,1]
    dkx = 2*np.pi/(config['dx']*config['N'])
    dky = 2*np.pi/(config['dx']*config['N'])
    return kx,ky,dkx,dky
