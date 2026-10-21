import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from util import *
from config import args

def A_operator(signal, Theta, num_samples, pupil, config):
    measurements_estimate = np.zeros((num_samples, config['N'], config['N']), dtype=np.complex128)

    kx,ky,dkx,dky = kspace_parameterize(Theta, config)
    for i in range(num_samples):
        lx,ly,hx,hy = find_bounds(kx[i].item(),ky[i].item(),dkx,dky,config)
        measurements_estimate[i,:,:] = iF(signal[ly:hy,lx:hx] * pupil)

    return measurements_estimate

def A_inverse_operator(measurements, Theta, pupil, config):
    Fs = np.fft.fftshift(np.fft.fft2(measurements), axes=(-2,-1))
    high_res_spectrums = np.zeros((measurements.shape[0],
                                   config['out_N'], config['out_N']),
                                  dtype=np.complex128)
    kx,ky,dkx,dky = kspace_parameterize(Theta, config)

    for i in range(Fs.shape[0]):
        lx,ly,hx,hy = find_bounds(kx[i].item(),ky[i].item(),dkx,dky,config)
        high_res_spectrums[i, ly:hy, lx:hx] = Fs[i,:,:] * np.conj(pupil)

    high_res_estimate = np.sum(high_res_spectrums, axis=0) / measurements.shape[0]
    return high_res_estimate

def reconstruction(observations, Theta, num_epochs, config):
    num_samples = observations.shape[0]
    pupil = config['M'].detach().cpu().numpy().astype(np.complex128)

    guess = np.sum(observations, axis=0) / num_samples
    guess = cv2.resize(guess, (config['out_N'], config['out_N']))
    guess_F = F(guess)

    GT = cv2.imread('/Users/cjnygard/UMD/RESEARCH/wavelength-fp/usaf_perfect_square.jpg', cv2.IMREAD_GRAYSCALE)
    GT = cv2.resize(GT, guess_F.shape)
    error = np.zeros(args.epochs + 1)
    error[0] = np.sum(np.abs(np.abs(guess_F)-np.abs(GT))) / np.sum(np.abs(GT))

    noise = np.zeros_like(observations)
    epsilon = noise

    normest = np.sqrt(np.sum(observations) / np.prod(observations.shape))
    step_size_noise = 0.01

    sigma2 = (observations.max() * 0.004) ** 2

    prog_bar = tqdm(range(num_epochs))
    first = True
    last_w = np.copy(guess_F)
    for i in prog_bar:
        Bz = A_operator(guess_F, Theta, num_samples, pupil, config)
        if Bz.max() > 100:
            import pdb;pdb.set_trace()
        Cz = (np.abs(Bz)**2 + noise - observations) * Bz
        w = A_inverse_operator(Cz, Theta, pupil, config)

        alpha = -np.log(0.997)
        mu = 1-np.exp(-alpha * (i+1))
        mu = np.minimum(mu, 0.4) / normest**2
        guess_F = guess_F - mu * w

        CN = (np.abs(Bz)**2 + noise - observations) + 1 * (noise * noise - 9 * sigma2 + epsilon * epsilon) * 2 * noise
        step_noise = mu * step_size_noise
        noise = noise - step_noise * CN

        epsilon_temp = 9 * sigma2 - noise * noise
        epsilon_temp[epsilon_temp < 0] = 0
        epsilon = np.sqrt(epsilon_temp)

        error[i+1] = np.sum(np.abs(np.abs(guess_F) - np.abs(GT))) / np.sum(np.abs(GT))
    print(error)
    print(error.min())
    print(np.argmin(error))
    return iF(guess_F)

def sim(observations, Theta, num_epochs, config):
    return reconstruction(observations ** 2, Theta, num_epochs, config)

def real(observations, Theta, num_epochs, config):
    return reconstruction(observations ** 2, Theta, num_epochs, config)

