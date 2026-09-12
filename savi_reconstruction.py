import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from util import *
from config import args
from multi_wavelength import *

def reconstruction(observations, Theta, num_epochs, config):
    num_samples = observations.shape[0]
    num_angles = Theta.shape[0]
 
    print(f"Number of observations: {num_samples}, Number of angles: {num_angles}")

    # guess = np.zeros((observations.shape[1], observations.shape[2]))
    # guess = np.copy(observations[7,:,:])
    guess = np.sum(observations, axis=0) / observations.shape[0]
    print(guess.shape)
    guess = cv2.resize(guess,
                       (guess.shape[0] * config['upsample'],
                        guess.shape[1] * config['upsample']))
    guessFT = F(guess)

    region = np.zeros(guessFT.shape, dtype=np.complex128)

    # create kx/ky for all three wavelengths
    kx = Theta[:,1]
    ky = Theta[:,0]
    dkx = 1
    dky = 1
    mask = config['M'].detach().cpu().numpy()
    pupil = np.where(mask, 1.0, 0.0).astype(np.complex128)
    og_pupil = np.copy(pupil)

    prog_bar = tqdm(range(num_epochs * num_samples))
    first = True
    for m in prog_bar:
        i = m % num_samples
        # CJCJCJ take measurement and deconstruct into the three
        # wavelengths. Run process three times per sample (once per sample
        # per wavelength) after deconstruction, wavelengths become separate
        # samples.
        if args.mode == 'traditional':
            measurements = np.expand_dims(observations[i], axis=-1)
        else:
            measurements = deconstruct_measurement_wavelengths(observations[i,:,:],
                                                               kx[i].item(),ky[i].item(), # need to figure this out
                                                               config)
        for channel in range(measurements.shape[2]):
            # adjust according to above kx/ky calculation
            lx,ly,hx,hy = find_bounds(kx[i].item(),ky[i].item(),dkx,dky,config)
            # print(f'{lx} {hx} , {ly} {hy}')
            r = guessFT[ly:hy, lx:hx]
            r_filtered = r * pupil
            low_res_guess = iF(r_filtered)

            # measurement = config['upsample']**2 * measurements[:,:,channel]

            # low_res_guess_new = np.sqrt(measurement) * np.exp(1j*np.angle(low_res_guess)) # replace amplitude but preserve phase
            # low_res_guess_FT = F(low_res_guess_new)

            measurement = measurements[:,:,channel] * args.upsample**2

            normal_low_res_guess = low_res_guess / np.abs(low_res_guess)
            magnitude_constrained = np.sqrt(measurement) * normal_low_res_guess
            magnitude_constrained_F = F(magnitude_constrained)

            pupil_step_size = np.abs(pupil) / np.max(np.abs(pupil))
            regularized_pupil_inverse = np.conj(pupil) / (np.abs(pupil)**2 + 1e-10)
            new_info = magnitude_constrained_F - r_filtered
            guessFT[ly:hy,lx:hx] = r +      \
                args.learning *             \
                pupil_step_size *           \
                regularized_pupil_inverse * \
                new_info

            measurement_step_size = np.abs(r) / np.max(np.abs(r))
            regularized_region_inverse = np.conj(r) / (np.abs(r)**2 + 1e-10)
            pupil = pupil +                     \
                args.learning *                 \
                measurement_step_size *         \
                regularized_region_inverse *    \
                new_info * og_pupil
            pupil = pupil * og_pupil

            if args.debug and i == m:
                region[ly:hy,lx:hx] =           \
                    region[ly:hy,lx:hx] +       \
                    args.learning *             \
                    pupil_step_size *           \
                    regularized_pupil_inverse * \
                    new_info
                # black = np.zeros(region.shape, dtype=np.complex128)
                # black[ly:hy,lx:hx] =     \
                #     pupil_step_size *           \
                #     regularized_pupil_inverse * \
                #     new_info
                guess = iF(guessFT)
                img1 = f'{prog_bar.n:03}_{i:03}_Measurement', observations[i,:,:]
                # tmp = np.abs(np.sqrt(np.log(F(observations[i,:,:]))))
                # img2 = f'{prog_bar.n:03}_{i:03}_', tmp - mask
                img2 = f'{prog_bar.n:03}_{i:03}_Guess', np.abs(guess)
                img3 = f'{prog_bar.n:03}_{i:03}_GuessF', np.abs(np.sqrt(np.log(1+guessFT)))
                # img3 = f'{prog_bar.n:03}_{i:03}_Spectrum', np.abs(np.sqrt(np.log(region)))
                # img4 = f'{prog_bar.n:03}_{i:03}_Mask', normalMask
                img4 = f'{prog_bar.n:03}_{i:03}_Region', np.abs(np.log(1+region))
                img5 = f'{prog_bar.n:03}_{i:03}_New', np.abs(np.sqrt(np.log(1+new_info)))
                img6 = f'{prog_bar.n:03}_{i:03}_Acutally_Adding', np.abs(np.sqrt(np.log(1+ pupil_step_size * regularized_pupil_inverse * new_info)))
                # img7 = f'black', np.abs(np.log(iF(black)))
                debug_print(img1,img2,img3,img4,img5,img6,save=False)
                # name_thing = f'{args.data_dir}/{args.epochs}e_{args.resolution}r_{args.learning}lr_i{i}'
                # cv2.imwrite(f'{name_thing}reconstruction_mid.jpg', np.abs(guess)*255)

    return iF(guessFT)

def sim(observations, Theta, num_epochs, config):
    return reconstruction(observations, Theta, num_epochs, config)

def real(observations, Theta, num_epochs, config):
    return reconstruction(observations, Theta, num_epochs, config)

