import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from tqdm import tqdm

import matlab_reconstruction
import ml_reconstruction
import savi_reconstruction
import wirtinger_flow_reconstruction
from config import *
from config import args
from forward_model import *
from load_data import *
from util import *

# device = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

# ===== Real data =====
if __name__ == "__main__":
    data_loc = args.data_dir
    if args.source == "umd_pi":
        samples_np, Theta, real_config = load_umd_pi_images(data_loc)
    elif args.source == "umd_bench":
        samples_np, Theta, real_config = load_umd_bench_images(data_loc)
    elif args.source == "array_sim":
        samples_np, Theta, real_config = load_array_sim_images(data_loc)
    elif args.source == "matlab":
        samples_np, Theta, real_config = load_matlab_images(data_loc)
    elif args.source == "waller":
        samples_np, Theta, real_config = load_waller_images(data_loc)
    elif args.source == "sim":
        samples_np, Theta, real_config = load_sim_images(data_loc)
    else:
        print(f'Unknown data source {args.source}')
        exit(1)

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
    # luh = [7,6,8,5,9,4,10,3,11,2,12,1,13,0,14]

    # pred = matlab_reconstruction.real(
    #     samples_np, Theta, num_epochs=args.epochs, config=real_config
    # )
    # pred = wirtinger_flow_reconstruction.real(
    #     samples_np, Theta, num_epochs=args.epochs, config=real_config
    # )
    pred = savi_reconstruction.real(
        samples_np, Theta, num_epochs=args.epochs, config=real_config
    )
    print(f"Samples shape {samples_np[0,:,:].shape}, Result shape {pred.shape}")
    if args.save:
        name_thing = f'{args.data_dir}/{args.epochs}e_{args.resolution}r_{args.learning}lr'
        abs = np.abs(pred)
        angle = np.angle(pred)
        cv2.imwrite(f'{name_thing}amplitude.jpg', (abs/np.max(abs))*255)
        # plt.imsave(f'{name_thing}phase.jpg', (angle/np.max(angle))*255, cmap='inferno')
        cv2.imwrite(f'{name_thing}phase.jpg', (angle/np.max(angle))*255)
        print(f'Results saved to {name_thing}amplitdue.jpg and {name_thing}phase.jpg')

    # W = args.resolution
    # gt = cv2.imread(f'{data_loc}/{args.data_name}{args.file_extension}', 0).astype(np.float32)
    # gt = cv2.resize(gt, (W*args.upsample,
    #                      W*args.upsample)) / 255.0
    #
    # from skimage.metrics import peak_signal_noise_ratio
    # psnr_value = peak_signal_noise_ratio(gt, np.real(pred))
    # print(f"PSNR: {psnr_value:.2f} dB")
    # psnr_value = peak_signal_noise_ratio(gt, cv2.resize(samples_np[0,:,:],
    #                                                     (W*args.upsample,
    #                                                      W*args.upsample)))
    # print(f"Center PSNR: {psnr_value:.2f} dB")

    img1 = 'First Measurement', samples_np[0,:,:]
    # img1 = 'Middle Measurement', samples_np[samples_np.shape[0]//2,:,:]
    # img1 = "Averaged", avg
    img2 = "Amplitude", np.abs(pred)
    img3 = "Phase", np.angle(pred)
    debug_print(img1, img2, img3)
