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

    if args.observations:
        samples_per = 40
        shape = 5, 8
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
        luh = [
            0,
            1,
            5,
            10,
            18,
            27,
            39,
            52,
            68,
            85,
            105,
            126,
            150,
            175,
            203,
        ]  # only horizontal
        # luh = [0, 1, 5, 10, 18]
        pred = matlab_reconstruction.real(
            samples_np, Theta, num_epochs=args.epochs, config=real_config
        )
        print(f"Samples shape {samples_np[0,:,:].shape}, Result shape {pred.shape}")
        # img1 = 'First Measurement', samples_np[0,:,:]
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

    LED_dir = "Data/USAF_red_LED_locations"
    LED_raw = scipy.io.loadmat(LED_dir)
    xlocation, ylocation = torch.from_numpy(
        LED_raw["xlocation_s"].astype(float)
    ), torch.from_numpy(LED_raw["ylocation_s"].astype(float))
    positions = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    Theta = LED_angles(positions, sim_config)

    num_samples = Theta.shape[0]
    observations = np.empty((num_samples, W // DOWNSAMPLE, W // DOWNSAMPLE))
    kx, ky, dkx, dky = kspace_parameterize(Theta, sim_config)
    O = F(sample)

    for i in range(num_samples):
        lx, ly, hx, hy = find_bounds(kx[i].item(), ky[i].item(), dkx, dky, sim_config)
        mask = sim_config["M"]
        mask = np.where(mask, 1.0, 0.0)
        observations[i, :, :] = np.abs(iF(O[ly:hy, lx:hx] * mask)) ** 2

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
        img1 = "Ground Truth", sample
        img2 = "Amplitude", np.abs(pred)
        img3 = "Phase", np.angle(pred)
        debug_print(img1, img2, img3)
    else:
        pred = ml_reconstruction.sim(
            sample, Theta, num_epochs=args.epochs, config=sim_config
        )
        img1 = "Measurement", forward_model(sample, torch.tensor([0, 0]), sim_config)
        img2 = "FP Reconstruction", pred.detach().cpu()
        img3 = "GT", sample
        debug_print(img1, img2, img3)
