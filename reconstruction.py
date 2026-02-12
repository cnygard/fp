import torch
import numpy as np
import scipy
from forward_model import *
from util import *
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from config import *
import ml_reconstruction
import matlab_reconstruction

isSim = False
epochs = 2
ourImages = True
if ourImages:
    data_name = 'Pumpkin'
    data_name = 'Heart'
    data_name = 'Stupid'
    data_name = 'Intestine'
    data_name = 'Crevice'
else:
    data_name = 'MouseKidney_green'
    data_name = 'USAF_red'
remove24 = False # remove bright field images
showObservations = False
cj = True # use CJ's reconstruction from Matlab code

print(f"isSim: {isSim}")
if __name__ == "__main__" and isSim:
    # grid_size = 5
    # spacing = 0.2
    # k_range = torch.linspace(-(grid_size // 2), grid_size // 2, grid_size) * spacing
    # kx, ky = torch.meshgrid(k_range, k_range, indexing="ij")
    # k_vectors = torch.stack((kx.flatten(), ky.flatten()), dim=1)
    # print(k_vectors)
    # === Load and preprocess image ===
    sample = cv2.imread("./Data/usaf.jpg", 0).astype(np.float32)
    sample = cv2.resize(sample, (256, 256))
    sample_np = sample / 255.0  # normalize
    sample = torch.from_numpy(sample_np)
    # region = F(sample)
    # cv2.imwrite(f'../real.jpg', np.log(np.real(region)) * 255.0)

    # sim_config = setup_config(sample, FPM_CONFIG)
    sim_config = setup_config(sample, UCONN_CONFIG)

    # === Setup angles ===
    theta_max = np.rad2deg(np.arcsin(sim_config["NA_illum"]))
    print(f"Max rotation (deg) {theta_max:.2e}")
    theta = torch.linspace(-theta_max, theta_max, 10)
    theta_y, theta_x = torch.meshgrid(theta, theta, indexing="ij")
    Theta = torch.stack([theta_y, theta_x], dim=-1).reshape(-1, 2)

    # CJCJCJ
    LED_dir = 'Data/'+data_name+'_LED_locations'
    LED_raw = scipy.io.loadmat(LED_dir)
    print(LED_raw.keys())
    xlocation, ylocation = torch.from_numpy(LED_raw['xlocation_s'].astype(float)), torch.from_numpy(LED_raw['ylocation_s'].astype(float))
    print(xlocation.shape, ylocation.shape)
    positions = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    sim_config['LED_spacing'] = 4 * 1e-3
    # CJCJCJ

    # print(positions)
    Theta = LED_angles(positions, sim_config)

    # print(f"Theta: {Theta}")

    if cj:
        # Theta = torch.deg2rad(Theta)
        pred = matlab_reconstruction.sim(sample_np, Theta,
                                      num_epochs=epochs,
                                      config=sim_config)
        print(f'Sample shape {sample.shape}, Result shape {pred.shape}')
        img1 = 'Ground Truth', sample
        img2 = 'Amplitude', np.abs(pred)
        img3 = 'Phase', np.angle(pred)
        debug_print(img1,img2,img3)
    else:
        pred = ml_reconstruction.sim(sample, Theta, num_epochs=epochs, config=sim_config)
        img1 = 'Measurement', forward_model(sample, torch.tensor([0, 0]), sim_config)
        img2 = 'FP Reconstruction', pred.detach().cpu()
        img3 = 'GT', sample
        debug_print(img1,img2,img3)
elif __name__ == "__main__" and not isSim:
    data_dir = 'Data/'+data_name
    LED_dir = 'Data/'+data_name+'_LED_locations'
    if ourImages:
        N = 256
        size = 616 // 2
        observations_raw = np.empty((N,size,size))
        for i in range(N):
            # img = cv2.imread(f'{data_dir}/test{i:03d}.jpg')/255.0 # read as BGR
            raw = np.load(f'{data_dir}/{i:03d}.npy')
            red = raw[0::2, 0::2]
            img = (red >> 2).astype(np.uint8)/255.0
            if img.shape[0] != img.shape[1]:
                # left = int((np.max(img[:,:,0].shape) - np.min(img[:,:,0].shape)) / 2)
                # right = left + np.min(img[:,:,0].shape)
                # observations_raw[i,:,:] = img[:,left:right,1] # green
                left = int((np.max(img.shape) - np.min(img.shape)) / 2)
                right = left + np.min(img.shape)
                observations_raw[i,:,:] = cv2.resize(img[:,left:right],
                                                     (size,size))
            else:
                observations_raw[i,:,:] = img
                # observations_raw[i,:,:] = img[:,:,1] # green

        size = observations_raw.shape[1]
        samples_np = observations_raw

        real_config = setup_config(samples_np[0,:,:], UCONN_CONFIG)

        width = np.sqrt(N)
        center = 7
        location = np.arange(width)
        xlocation = location - center
        ylocation = xlocation
        x,y = np.meshgrid(xlocation, ylocation, indexing='xy')
        positions = np.stack([y,x], axis=-1).reshape(-1,2)
        Theta = LED_angles(torch.from_numpy(positions), real_config)
        print(f"Samples shape: {samples_np.shape} Theta shape: {Theta.shape}")
        order = spiral((7,7),16)
        Theta = rearrange(Theta, order)
        # print(Theta)
        # import pdb;pdb.set_trace()
        # samples_np = rearrange(samples_np, order)
        if False:
            for slide in range(int(np.ceil(samples_np.shape[0]/40))):
                for i in range(40):
                    if i+slide*40 > samples_np.shape[0]-1:
                        break
                    plt.subplot(5,8,i+1)
                    plt.title('raw image '+str(order[i+slide*40]))
                    plt.imshow(samples_np[i+slide*40,:,:],cmap='gray')
                    plt.axis('off')
                plt.show()
    else:
        # === Load and preprocess image ===
        observations_raw = scipy.io.loadmat(data_dir)['imlow_HDR'].transpose(2,0,1) # first index is now image index

        # === Setup observations ===
        N = observations_raw.shape[0]
        size = observations_raw.shape[1]
        samples_np = np.zeros((N, size, size), dtype=np.float32)
        for i in range(N):
            samples_np[i,:,:] = cv2.resize(observations_raw[i,:,:], (size, size)) / 255.0
        samples = torch.from_numpy(samples_np)
        if remove24:
            samples = samples[24:,:,:] # filter out bright field images

        if showObservations:
            for slide in range(int(np.ceil(samples_np.shape[0]/40))):
                for i in range(40):
                    if i+slide*40 > samples_np.shape[0]-1:
                        break
                    plt.subplot(5,8,i+1)
                    plt.title('raw image '+str(i+slide*40+1))
                    plt.imshow(samples_np[i+slide*40,:,:],cmap='gray')
                    plt.axis('off')
                plt.show()

        real_config = setup_config(samples[0,:,:], UCONN_CONFIG)

        # === Load LED locations ===
        LED_raw = scipy.io.loadmat(LED_dir)
        print(LED_raw.keys())
        xlocation, ylocation = torch.from_numpy(LED_raw['xlocation_s'].astype(float)), torch.from_numpy(LED_raw['ylocation_s'].astype(float))
        print(xlocation.shape, ylocation.shape)
        positions = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
        Theta = LED_angles(positions, real_config)
        if remove24:
            Theta = Theta[24:,:] # filter out bright field angles
        print(f"Samples shape: {samples.shape} Theta shape: {Theta.shape}")

    # # visualize a few real observations
    # img1 = 'Measurement 0', samples[0,:,:]
    # img2 = 'Measurement 100', samples[100,:,:]
    # img3 = 'Measurement 200', samples[200,:,:]
    # debug_print(img1,img2,img3)

    if cj:
        pred = matlab_reconstruction.real(samples_np, Theta,
                                          num_epochs=epochs,
                                          config=real_config)
        print(np.max(np.abs(pred)))
        print(f'Samples shape {samples_np[0,:,:].shape}, Result shape {pred.shape}')
        img1 = 'First Measurement', samples_np[0,:,:]
        img2 = 'Amplitude', np.abs(pred)
        img3 = 'Phase', np.angle(pred)
        debug_print(img1,img2,img3)
    else:
        pred = ml_reconstruction.real(samples, Theta, num_epochs=epochs, config=real_config)
        print(f'Samples shape {samples[0,:,:].shape}, Result shape {pred.shape}')
        img1 = 'First Measurement', samples[0,:,:]
        img2 = 'FP Reconstruction', pred.detach().cpu().numpy().astype('float')
        # img3 = 'GT', sample
        debug_print(img1,img2)
