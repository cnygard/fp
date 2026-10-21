import cv2
import numpy as np
import pywt
from tqdm import tqdm

from config import args
from multi_wavelength import *
from util import *


def reconstruction(observations, Theta, num_epochs, config):
    denoising_frequency = 10

    num_samples = observations.shape[0]
    num_angles = Theta.shape[0]

    print(f"Number of observations: {num_samples}, Number of angles: {num_angles}")

    # guess = np.zeros((observations.shape[1], observations.shape[2]))
    # guess = np.copy(observations[7,:,:])
    guess = np.sum(observations, axis=0) / observations.shape[0]
    guess = cv2.resize(
        guess,
        (guess.shape[0] * config["upsample"], guess.shape[1] * config["upsample"]),
    )
    guessFT = F(guess)

    region = np.zeros(guessFT.shape)

    kx, ky, dkx, dky = kspace_parameterize(Theta, config)
    mask = config["M"].detach().cpu().numpy()
    mask = np.where(mask, 1.0, 0.0)
    pupil_support = mask.copy()

    prog_bar = tqdm(range(num_epochs))
    first = True
    for epoch in prog_bar:
        for i in range(num_samples):
            lx, ly, hx, hy = find_bounds(kx[i].item(), ky[i].item(), dkx, dky, config)
            # print(f'{lx} {hx} , {ly} {hy}')
            r = guessFT[ly:hy, lx:hx].copy()
            r_masked = r * mask
            low_res_guess = iF(r_masked)

            amplitude = config["upsample"] ** 2 * np.sqrt(observations[i, :, :])

            # measurement_F = np.fft.fftshift(np.fft.fft2(measurement))
            # def make_viewable(img):
            #     return np.abs(np.log(1+img))
            # img1 = f'measurement', measurement
            # img2 = f'F', make_viewable(measurement_F)
            # img3 = f'F_masked', make_viewable(measurement_F * mask)
            # img4 = f'iF_F_masked', np.abs(np.fft.ifft2(np.fft.ifftshift(measurement_F * mask)))
            # debug_print(img1,img2,img3,img4)

            # low_res_guess_new = np.sqrt(measurement) * np.exp(1j*np.angle(low_res_guess)) # replace amplitude but preserve phase
            # low_res_guess_new = amplitude * low_res_guess / np.abs(low_res_guess)
            low_res_guess_new = amplitude * np.exp(1j * np.angle(low_res_guess))
            low_res_guess_FT = F(low_res_guess_new)

            current_guess_max = np.abs(guessFT).max()

            tau = 1e-10  # Tiny non-zero number for div by 0
            mask_step_size = np.abs(mask) / np.abs(mask).max()
            normal_mask = np.conj(mask) / (np.abs(mask) ** 2 + tau)
            # guess_step_size = np.abs() / np.abs(mask).max()
            new_info = low_res_guess_FT - r_masked
            guessFT[ly:hy, lx:hx] = (
                guessFT[ly:hy, lx:hx] + mask_step_size * normal_mask * new_info
            )

            # r_step_size = np.abs(r) / current_guess_max
            # normal_r = np.conj(r) / (np.abs(r)**2 + tau)
            # pupil_update = r_step_size * normal_r * new_info
            # mask = (mask + pupil_update) * pupil_support

            normalMask = np.conj(mask) / np.max((np.abs(mask)) ** 2)
            newInfo = low_res_guess_FT - r_masked
            guessFT[ly:hy, lx:hx] = (
                guessFT[ly:hy, lx:hx] + args.learning * normalMask * newInfo
            )

            if args.debug and first:
                region[ly:hy, lx:hx] = (
                    region[ly:hy, lx:hx] + mask_step_size * normal_mask * new_info
                )
                # black = np.zeros(region.shape)
                # black[ly:hy,lx:hx] =     \
                #     normal_mask *         \
                #     new_info
                guess = iF(guessFT)
                img1 = f"{prog_bar.n:03}_{i:03}_Measurement", observations[i, :, :]
                # tmp = np.abs(np.sqrt(np.log(F(observations[i,:,:]))))
                # img2 = f'{prog_bar.n:03}_{i:03}_', tmp - mask
                img2 = f"{prog_bar.n:03}_{i:03}_Guess", np.abs(guess)
                img3 = (
                    f"{prog_bar.n:03}_{i:03}_GuessF",
                    np.abs(np.sqrt(np.log(guessFT))),
                )
                # img3 = f'{prog_bar.n:03}_{i:03}_Spectrum', np.abs(np.sqrt(np.log(region)))
                # img4 = f'{prog_bar.n:03}_{i:03}_Mask', normalMask
                img4 = f"{prog_bar.n:03}_{i:03}_Region", np.abs(np.log(region))
                img5 = f"{prog_bar.n:03}_{i:03}_New", np.abs(np.sqrt(np.log(new_info)))
                # img6 = f'{prog_bar.n:03}_{i:03}_Acutally_Adding', np.abs(np.sqrt(np.log(normal_mask * new_info)))
                img6 = f"{prog_bar.n:03}_{i:03}_Mask", np.abs(mask)
                # img7 = f'black', np.abs(np.log(iF(black)))
                debug_print(img1, img2, img3, img4, img5, img6, save=False)
        # if (epoch + 1) % denoising_frequency == 0:
        #     guess = iF(guessFT)
        #     intensity = np.abs(guess ** 2)
        #     xi = np.log(np.abs(guess) ** 2)
        #     coeffs = pywt.wavedec2(xi, wavelet="sym8", level=2)
        #     rho = 1
        #     xi_denoised = np.sign(coeffs) * np.maximum(np.abs(coeffs) - rho, 0)
        #     guess = np.exp(0.5 * xi_denoised) * np.exp(1j * np.angle(guess))
        #     guessFT = F(guess)
        first = False

    return iF(guessFT)


def sim(observations, Theta, num_epochs, config):
    return reconstruction(observations, Theta, num_epochs, config)


def real(observations, Theta, num_epochs, config):
    return reconstruction(observations, Theta, num_epochs, config)
