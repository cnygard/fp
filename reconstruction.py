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

# ===== Real data =====
if __name__ == "__main__":
    data_loc = args.data_dir
    if args.source == "umd_pi":
        samples_np, Theta, real_config = load_umd_pi_images(data_loc)
    elif args.source == "umd_bench":
        samples_np, Theta, real_config = load_umd_bench_images(data_loc)
    elif args.source == "matlab":
        samples_np, Theta, real_config = load_matlab_images(data_loc)
    elif args.source == "waller":
        samples_np, Theta, real_config = load_waller_images(data_loc)
    elif args.source == "sim":
        samples_np, Theta, real_config = load_sim_images(data_loc)

    if args.observations:
        samples_per = 20
        shape = 3, 5
        for slide in range(int(np.ceil(samples_np.shape[0] / samples_per))):
            for i in range(samples_per):
                if i + slide * samples_per > samples_np.shape[0] - 1:
                    break
                plt.subplot(shape[0], shape[1], i + 1)
                plt.title("raw image " + str(i + slide * samples_per))
                plt.imshow(samples_np[i + slide * samples_per, :, :], cmap="gray")
                plt.axis("off")
            # plt.tight_layout()
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            plt.show()

    # calc resized average
    upy = samples_np.shape[1] * real_config['upsample']
    upx = samples_np.shape[2] * real_config['upsample']
    samples_resized = np.empty((samples_np.shape[0], upy, upx))
    for i in range(samples_resized.shape[0]):
        samples_resized[i,:,:] = cv2.resize(samples_np[i,:,:], (upy,upx))
    avg = np.sum(samples_resized, axis=0) / samples_resized.shape[0]
    # avg = np.sum(samples_np, axis=0) / samples_np.shape[0]

    # luh = [14,12,10,8,6,4,2,0,1,3,5,7,9,11,13] # move inside to outside

    pred = matlab_reconstruction.real(
        samples_np, Theta, num_epochs=args.epochs, config=real_config
    )
    print(f"Samples shape {samples_np[0,:,:].shape}, Result shape {pred.shape}")
    if args.save:
        cv2.imwrite(f'{args.data_dir}/{args.data_name}_amplitude{args.file_extension}', np.abs(pred)*255)
        cv2.imwrite(f'{args.data_dir}/{args.data_name}_phase{args.file_extension}', np.angle(pred)*255)
        print('Results saved to Data/output_amplitdue.jpg and Data/output_phase.jpg')
    # img1 = 'First Measurement', samples_np[0,:,:]
    # img1 = 'Middle Measurement', samples_np[samples_np.shape[0]//2,:,:]
    img1 = "Averaged", avg
    img2 = "Amplitude", np.abs(pred)
    img3 = "Phase", np.angle(pred)
    debug_print(img1, img2, img3)

# # ===== Simulation =====
# elif __name__ == "__main__" and args.sim:
#     # === Load and preprocess image ===
#     W = args.resolution
#     data_dir = "Data/" + args.data_name
#     sample = cv2.imread(data_dir, 0).astype(np.float32)
#     sim_config = setup_config(
#         torch.zeros(W, W), UMD_SLIT_CONFIG
#     )
#     sample = cv2.resize(sample, (W*sim_config['upsample'],
#                                  W*sim_config['upsample'])) / 255.0
#
#     N = args.samples_count
#     observations = np.empty((N, W, W))
#
#     if args.mode == "translate":
#         center = N // 2
#         location = np.arange(N)
#         xlocation = location - center
#         ylocation = np.zeros_like(xlocation)
#         positions = np.stack([xlocation,ylocation], axis=-1).reshape(-1,2)
#         shift = positions * -1 * sim_config['translation']
#         mask = sim_config['M'].detach().cpu().numpy()
#         mask = np.where(mask, 1.0, 0.0)
#     elif args.mode == "rotate":
#         # TODO fix later, hardcoded old values for now
#         shift = np.array([[-90, 0], [-67.5, 0], [-45, 0], [-22.5, 0], [0, 0], [22.5, 0], [45, 0], [67.5, 0]])
#         # unfinished, pass rotation angles as Theta?
#         masks = np.zeros((N, sim_config['N'], sim_config['N']))
#         for i in range(N):
#             masks[i] = rotating_bar_mask((sim_config['N'], sim_config['N']), shift[i,0], args.slit_width)
#
#     if args.mode == "translate":
#         kx = shift[:,0]
#         ky = shift[:,1]
#         dkx = 1
#         dky = 1
#     elif args.mode == "rotate":
#         kx = np.zeros_like(shift[:,0])
#         ky = np.zeros_like(shift[:,1])
#         dkx = 1
#         dky = 1
#
#     O = F(sample)
#
#     for i in range(N):
#         lx, ly, hx, hy = find_bounds(kx[i].item(), ky[i].item(), dkx, dky, sim_config)
#         # mask = sim_config['Ms'][i]
#         # mask = np.where(mask, 1.0, 0.0)
#         if args.mode == "translate":
#             masked = O[ly:hy, lx:hx] * mask
#         elif args.mode == "rotate":
#             masked = O[ly:hy, lx:hx] * masks[i]
#         observations[i, :, :] = np.abs(iF(masked))
#         # observations[i, :, :] = np.abs(iF(masked)) ** 2
#         # plt.subplot(1,2,1)
#         # plt.imshow(observations[i])
#         # plt.subplot(1,2,2)
#         # plt.imshow(np.abs(np.log(masked)))
#         # plt.show()
#
#     if args.observations:
#         samples_per = 12
#         shape = 3, 4
#         for slide in range(int(np.ceil(observations.shape[0] / samples_per))):
#             for i in range(samples_per):
#                 if i + slide * samples_per > observations.shape[0] - 1:
#                     break
#                 plt.subplot(shape[0], shape[1], i + 1)
#                 plt.title("raw image " + str(i + slide * samples_per))
#                 plt.imshow(
#                     observations[i + slide * samples_per, :, :],
#                     cmap="hot",
#                 )
#                 plt.axis("off")
#             plt.tight_layout()
#             plt.show()
#
#     if cj:
#         # Theta = torch.deg2rad(Theta)
#         pred = matlab_reconstruction.sim(
#                 observations, shift, num_epochs=args.epochs, config=sim_config
#         )
#         print(f"Sample shape {sample.shape}, Result shape {pred.shape}")
#         # img1 = "Ground Truth", sample
#         guess = np.sum(observations, axis=0) / observations.shape[0]
#         guess = cv2.resize(guess,
#                            (guess.shape[0] * sim_config['upsample'],
#                             guess.shape[1] * sim_config['upsample']))
#         img1 = "Sum Guess", guess
#         img2 = "Amplitude", np.abs(pred)
#         # img3 = "F", np.real(np.log(F(pred)))
#         img3 = "Phase", np.angle(pred)
#         debug_print(img1, img2, img3)
#     else:
#         pred = ml_reconstruction.sim(
#             sample, Theta, num_epochs=args.epochs, config=sim_config
#         )
#         img1 = "Measurement", forward_model(sample, torch.tensor([0, 0]), sim_config)
#         img2 = "FP Reconstruction", pred.detach().cpu()
#         img3 = "GT", sample
#         debug_print(img1, img2, img3)
