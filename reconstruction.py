import cv2
import matlab_reconstruction
import matplotlib.pyplot as plt
import ml_reconstruction
import numpy as np
import scipy
import torch
from config import *
from config import args
from forward_model import *
from load_data import *
from tqdm import tqdm

from util import *

# if data_source == 'umd_pi':
#     data_name = 'Pumpkin'
#     data_name = 'Heart'
#     data_name = 'Stupid'
#     data_name = 'Intestine'
#     data_name = 'Crevice'
#     data_name = 'PI_USAF_OLD'
#     data_name = 'PI_USAF_NEW'
# elif data_source == 'umd_bench':
#     # data_name = 'usaf_15_every20'
#     # data_name = 'usaf_15_every10'
#     # data_name = 'usaf_11_wide_slit'
#     # data_name = 'usaf_11_thin_slit'
#     # data_name = 'usaf_15_whole_turn'
#     # data_name = 'slit_tilt_5'
#     # data_name = 'slit_tilt_8'
#     data_name = 'usaf_15_every35_calc'
# elif data_source == 'matlab':
#     data_name = 'MouseKidney_green'
#     data_name = 'USAF_red'
# elif data_source == 'waller':
#     data_name = 'Waller'
#     # data_name = 'Waller-Full'

cj = True  # use CJ's reconstruction from Matlab code

print(f"Sim: {args.sim}")
# ===== Real data =====
if __name__ == "__main__" and not args.sim:
    data_dir = "Data/" + args.data_name
    if args.source == "umd_pi":
        samples_np, Theta, real_config = load_umd_pi_images(data_dir)
    elif args.source == "umd_bench":
        samples_np, Theta, real_config = load_umd_bench_images(data_dir)
    elif args.source == "matlab":
        samples_np, Theta, real_config = load_matlab_images(data_dir)
    elif args.source == "waller":
        samples_np, Theta, real_config = load_waller_images(data_dir)

    # plt.imshow(samples_np[0])
    # plt.show()
    # plt.imshow(samples_np[1])
    # plt.show()
    # plt.imshow(samples_np[2])
    # plt.show()
    if args.observations:
        samples_per = 20
        shape = 4, 5
        for slide in range(int(np.ceil(samples_np.shape[0] / samples_per))):
            for i in range(samples_per):
                if i + slide * samples_per > samples_np.shape[0] - 1:
                    break
                plt.subplot(shape[0], shape[1], i + 1)
                plt.title("raw image " + str(i + slide * samples_per))
                plt.imshow(samples_np[i + slide * samples_per, :, :], cmap="gray")
                plt.axis("off")
            plt.tight_layout()
            plt.show()

    if cj:
        # luh = np.r_[0, 9:24, 49:80, 121:168] # skip one layer for less overlap
        # luh = np.r_[0, 25:48, 121:168] # skip 2 layers for even less
        # luh = [0,1,5,10,18,27,39,52,68,85,105,126,150,175,203]  # only horizontal
        # luh = [0, 1, 5, 10, 18]
        luh = [14,12,10,8,6,4,2,0,1,3,5,7,9,11,13] # move inside to outside
        pred = matlab_reconstruction.real(
            samples_np, Theta, num_epochs=args.epochs, config=real_config
        )
        print(f"Samples shape {samples_np[0,:,:].shape}, Result shape {pred.shape}")
        if args.save:
            cv2.imwrite('Data/output_amplitude.jpg', np.abs(pred)*255)
            cv2.imwrite('Data/output_phase.jpg', np.angle(pred)*255)
            print('Results saved to Data/output_amplitdue.jpg and Data/output_phase.jpg')
        # img1 = 'First Measurement', samples_np[0,:,:]
        # img1 = 'Middle Measurement', samples_np[samples_np.shape[0]//2,:,:]
        img1 = "Averaged", np.sum(samples_np, axis=0) / samples_np.shape[0]
        img2 = "Amplitude", np.abs(pred)
        img3 = "Phase", np.angle(pred)
        debug_print(img1, img2, img3)
    else:
        pred = ml_reconstruction.real(
            samples, Theta, num_epochs=args.epochs, config=real_config
        )
        print(f"Samples shape {samples[0,:,:].shape}, Result shape {pred.shape}")
        img1 = "First Measurement", samples[0, :, :]
        img2 = "FP Reconstruction", pred.detach().cpu().numpy().astype("float")
        debug_print(img1, img2)


# ===== Simulation =====
elif __name__ == "__main__" and args.sim:
    # === Load and preprocess image ===
    W = 564
    DOWNSAMPLE = 4
    data_dir = "Data/" + args.data_name
    sample = cv2.imread(data_dir, 0).astype(np.float32)
    sample = cv2.resize(sample, (W, W)) / 255.0
    sim_config = setup_config(
        torch.zeros(W // DOWNSAMPLE, W // DOWNSAMPLE), UCONN_CONFIG
    )

    num_samples = 8
    observations = np.empty((num_samples, W // DOWNSAMPLE, W // DOWNSAMPLE))

    if args.mode == "translate":
        width = num_samples
        center = width // 2
        location = np.arange(width)
        xlocation = location - center
        ylocation = np.zeros_like(xlocation)
        positions = np.stack([xlocation,ylocation], axis=-1).reshape(-1,2)
        shift = positions * 150 * -1
        mask = sim_config['M'].detach().cpu().numpy()
        mask = np.where(mask, 1.0, 0.0)
    elif args.mode == "rotate":
        # TODO fix later, hardcoded old values for now
        shift = np.array([[-90, 0], [-67.5, 0], [-27, 0], [-17.5, 0], [1, 0], [12.5, 0], [30, 0], [67.5, 0]])
        # unfinished, pass rotation angles as Theta?
        masks = np.zeros((num_samples, sim_config['N'], sim_config['N']))
        for i in range(num_samples):
            masks[i] = rotating_bar_mask((config['N'], sim_config['N']), Theta[i,0], args.slit_width)

    kx = shift[:,0]
    ky = shift[:,1]
    dkx = 1
    dky = 1
    O = F(sample)

    for i in range(num_samples):
        lx, ly, hx, hy = find_bounds(kx[i].item(), ky[i].item(), dkx, dky, sim_config)
        # mask = sim_config['Ms'][i]
        # mask = np.where(mask, 1.0, 0.0)
        masked = O[ly:hy, lx:hx] * mask
        observations[i, :, :] = np.abs(iF(masked)) ** 2
        plt.subplot(1,2,1)
        plt.imshow(observations[i])
        plt.subplot(1,2,2)
        plt.imshow(np.abs(np.log(masked)))
        plt.show()

    if args.observations:
        samples_per = 12
        shape = 3, 4
        for slide in range(int(np.ceil(observations.shape[0] / samples_per))):
            for i in range(samples_per):
                if i + slide * samples_per > observations.shape[0] - 1:
                    break
                plt.subplot(shape[0], shape[1], i + 1)
                plt.title("raw image " + str(i + slide * samples_per))
                plt.imshow(
                    observations[i + slide * samples_per, :, :],
                    cmap="hot",
                )
                plt.axis("off")
            plt.tight_layout()
            plt.show()

    if cj:
        # Theta = torch.deg2rad(Theta)
        pred = matlab_reconstruction.sim(
                observations, Theta, num_epochs=args.epochs, config=sim_config
        )
        print(f"Sample shape {sample.shape}, Result shape {pred.shape}")
        # img1 = "Ground Truth", sample
        guess = np.sum(observations, axis=0) / observations.shape[0]
        guess = cv2.resize(guess,
                           (guess.shape[0] * sim_config['upsample'],
                            guess.shape[1] * sim_config['upsample']))
        img1 = "Sum Guess", guess
        img2 = "Amplitude", np.abs(pred)
        img3 = "F", np.real(np.log(F(pred)))
        # img3 = "Phase", np.angle(pred)
        debug_print(img1, img2, img3)
    else:
        pred = ml_reconstruction.sim(
            sample, Theta, num_epochs=args.epochs, config=sim_config
        )
        img1 = "Measurement", forward_model(sample, torch.tensor([0, 0]), sim_config)
        img2 = "FP Reconstruction", pred.detach().cpu()
        img3 = "GT", sample
        debug_print(img1, img2, img3)
