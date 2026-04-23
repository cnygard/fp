import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from util import *

debug = True
onlyMiddle = False
filterNorm = 0.25
learningScalar = 0.01

def reconstruction(observations, Theta, num_epochs, config):
    if onlyMiddle:
        middle_inds = torch.argwhere(torch.norm(Theta, dim=1) < filterNorm)
        # print(middle_inds)
        observations = observations[middle_inds.flatten(),:,:]
        Theta = Theta[middle_inds.flatten(),:]

    num_samples = observations.shape[0]
    num_angles = Theta.shape[0]
    
    print(f"Number of observations: {num_samples}, Number of angles: {num_angles}")

    # guess = np.zeros((observations.shape[1], observations.shape[2]))
    # guess = np.copy(observations[0,:,:])
    guess = np.sum(observations, axis=0) / observations.shape[0]
    
    print(guess.shape)
    # plt.imshow(guess,cmap='gray')
    # plt.show()

    guess = cv2.resize(guess,
                       (guess.shape[0] * config['upsample'],
                        guess.shape[1] * config['upsample']))
    guessFT = F(guess)

    # debug view
    # region = np.copy(guessFT)
    region = np.zeros(guessFT.shape)

    if config['title'] == 'UMD Bench Microscopy':
        kx = Theta[:,0]
        ky = Theta[:,1]
        dkx = 1
        dky = 1
    else:
        kx,ky,dkx,dky = kspace_parameterize(Theta, config)

    mask = config['M'].detach().cpu().numpy()
    mask = np.where(mask, 1.0, 0.0)
    blursize = 45
    # mask = cv2.GaussianBlur(mask, (blursize,blursize), 0)
    # plt.imshow(mask,cmap='gray')
    # plt.show()

    # need angle per measurement to reconstruct
    # call replace-angle on sample
    prog_bar = tqdm(range(2))
    first = True
    for _ in prog_bar:
        for i in range(num_samples):
            # image = sample # reproject current estimate to same angle as this observation
            lx,ly,hx,hy = find_bounds(kx[i].item(),ky[i].item(),dkx,dky,config)
            r = guessFT[ly:hy, lx:hx]
            r_masked = r * mask
            low_res_guess = iF(r_masked)
            measurement = config['upsample']**2 * observations[i,:,:]
            low_res_guess = measurement * np.exp(1j*np.angle(low_res_guess)) # replace amplitude but preserve phase
            low_res_guess_FT = F(low_res_guess)

            normalMask = np.conj(mask)/np.max((np.abs(mask))**2)
            newInfo = (low_res_guess_FT - r_masked)
            guessFT[ly:hy,lx:hx] =     \
                guessFT[ly:hy,lx:hx] + \
                learningScalar *       \
                normalMask *           \
                newInfo

            if debug and first:
                region[ly:hy,lx:hx] =     \
                    region[ly:hy,lx:hx] + \
                    normalMask *          \
                    newInfo
                black = np.zeros(region.shape)
                black[ly:hy,lx:hx] =     \
                    normalMask *         \
                    newInfo
                guess = iF(guessFT)
                img1 = f'{prog_bar.n:03}_{i:03}_Measurement', observations[i,:,:]
                img2 = f'{prog_bar.n:03}_{i:03}_Guess', np.abs(guess)
                img3 = f'{prog_bar.n:03}_{i:03}_Spectrum', np.abs(np.sqrt(np.log(region)))
                # img4 = f'{prog_bar.n:03}_{i:03}_Mask', normalMask
                img4 = f'{prog_bar.n:03}_{i:03}_Whole', np.abs(np.sqrt(np.log(guessFT)))
                img5 = f'{prog_bar.n:03}_{i:03}_New', np.abs(np.sqrt(np.log(newInfo)))
                img6 = f'{prog_bar.n:03}_{i:03}_Acutally_Adding', np.abs(np.sqrt(np.log(normalMask * newInfo)))
                img7 = f'black', np.abs(np.log(iF(black)))
                debug_print(img1,img2,img3,img4,img5,img6,save=False)
                # debug_print(img7,img1,img2,save=False)
        first = False
    
    # momentum
    countimg = 0
    tt = np.ones(num_epochs * num_samples)

    m, n = guessFT.shape
    vobj0 = np.zeros((m, n))
    vp0 = np.zeros_like(observations[0,:,:])
    ObjT = np.copy(guessFT)
    PT = np.copy(mask)

    gamma_obj = 1 # hardcoded all these since Matlab did too
    alpha = 1
    beta = 1
    T = 1
    eta_obj = 0.2 # is what Matlab uses
    eta_p = 0.2 # ^

    prog_bar = tqdm(range(num_epochs))
    for epoch in prog_bar:
        for i in range(num_samples):
            countimg += 1
            lx,ly,hx,hy = find_bounds(kx[i].item(),ky[i].item(),dkx,dky,config)
            r = guessFT[ly:hy, lx:hx]
            r_masked = r * mask
            low_res_guess = iF(r_masked)

            tt[i + (epoch * num_samples)] = (np.mean(np.abs(low_res_guess))) / (np.mean(config['upsample']**2 * np.abs(observations[i,:,:])))
            if epoch > 1:
                observations[i,:,:] = observations[i,:,:] * tt[i + (epoch * num_samples)]

            measurement = config['upsample']**2 * observations[i,:,:]
            low_res_guess = measurement * np.exp(1j*np.angle(low_res_guess)) # replace amplitude but preserve phase
            low_res_guess_FT = F(low_res_guess)

            normalMask = (np.conj(mask) * gamma_obj)/((1 - alpha) * np.abs(mask)**2 + alpha * np.max((np.abs(mask))**2))
            newInfo = (low_res_guess_FT - r_masked)
            guessFT[ly:hy,lx:hx] =     \
                guessFT[ly:hy,lx:hx] + \
                learningScalar *       \
                normalMask *           \
                newInfo
            normalR = (np.conj(r) * gamma_obj)/((1 - beta) * np.abs(r)**2 + beta * np.max((np.abs(r))**2)) # does r change?
            mask = mask + (newInfo * gamma_obj * normalR) # add learning scalar?

            if countimg == T: # momentum method
                vobj = eta_obj * vobj0 + (guessFT - ObjT)
                guessFT = ObjT + vobj
                vobj0 = vobj
                ObjT = np.copy(guessFT)

                vp = eta_p * vp0 + (mask - PT)
                mask = PT + vp
                vp0 = vp
                PT = np.copy(mask)

                countimg = 0
        # img1 = f'Best_Sample', observations[7]
        # img2 = f'Amplitude', np.abs(iF(guessFT))
        # img3 = f'Phase', np.angle(iF(guessFT))
        # debug_print(img1,img2,img3,save=False)
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
