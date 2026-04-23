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
epochs = 0

# data_source = 'umd_pi'
data_source = 'umd_bench'
# data_source = 'matlab'
# data_source = 'waller'

if data_source == 'umd_pi':
    data_name = 'Pumpkin'
    data_name = 'Heart'
    data_name = 'Stupid'
    data_name = 'Intestine'
    data_name = 'Crevice'
elif data_source == 'umd_bench':
    data_name = 'usaf_15_every20'
    data_name = 'usaf_15_every10'
    data_name = 'usaf_11_wide_slit'
    data_name = 'usaf_11_thin_slit'
    data_name = 'usaf_15_whole_turn'
    data_name = 'slit_tilt_5'
    data_name = 'slit_tilt_8'
elif data_source == 'matlab':
    data_name = 'MouseKidney_green'
    data_name = 'USAF_red'
elif data_source == 'waller':
    data_name = 'Waller'
    # data_name = 'Waller-Full'

remove24 = False # remove bright field images
showObservations = False
cj = True # use CJ's reconstruction from Matlab code

def load_umd_pi_images(data_dir):
    N = 256
    # size = 616 // 2
    size = 250
    observations_raw = np.empty((N,size,size))
    left,right,top,bottom = 225,475,175,425
    for i in range(N):
        # img = cv2.imread(f'{data_dir}/test{i:03d}.jpg')/255.0 # read as BGR
        raw = np.load(f'{data_dir}/{i:03d}.npy')
        # red = raw[0::2, 0::2]
        # img = (red >> 2).astype(np.uint8)/255.0
        G1 = raw[0::2, 1::2]
        G2 = raw[1::2, 0::2]
        G = ((G1.astype(np.uint32) + G2.astype(np.uint32)) // 2).astype(np.uint16)
        img = (G >> 2).astype(np.uint8)
        if True:
            observations_raw[i,:,:] = np.sqrt(img[top:bottom,left:right])
        elif img.shape[0] != img.shape[1]:
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
    order = spiral((center,center),16)
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
    return samples_np, Theta, real_config

def load_umd_bench_images(data_dir):
    N = 8
    # size = 1080
    size = 270
    observations_raw = np.empty((N,size,size))
    for i in range(N):
        print(f'{data_dir}/{i:04d}.jpg')
        img = cv2.imread(f'{data_dir}/{i:04d}.jpg', cv2.IMREAD_GRAYSCALE)/255.0
        # img = cv2.imread(f'{data_dir}/{i+1:04d}.jpg', cv2.IMREAD_GRAYSCALE)/255.0
        if img.shape[0] != img.shape[1]:
            left = int((np.max(img.shape) - np.min(img.shape)) / 2)
            right = left + np.min(img.shape)
            observations_raw[i,:,:] = cv2.resize(img[:,left:right],
                                                 (size,size))
        else:
            # observations_raw[i,:,:] = img
            observations_raw[i,:,:] = img

    size = observations_raw.shape[1]
    samples_np = observations_raw

    real_config = setup_config(samples_np[0,:,:], UMD_BENCH_CONFIG)

    width = N
    center = width // 2
    location = np.arange(width)
    xlocation = location - center
    ylocation = np.zeros_like(xlocation)
    positions = np.stack([xlocation,ylocation], axis=-1).reshape(-1,2)
    # shift = positions * 8
    shift = positions * 2
    # Theta = LED_angles(torch.from_numpy(positions), real_config)
    # print(f"Samples shape: {samples_np.shape} Theta shape: {Theta.shape}")
    # order = spiral((center,center),15)
    # Theta = rearrange(Theta, order)
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
    return samples_np, shift, real_config

def load_matlab_images(data_dir):
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
    xlocation, ylocation = (torch.from_numpy(LED_raw['xlocation_s'].astype(float)),
                            torch.from_numpy(LED_raw['ylocation_s'].astype(float)))
    print(xlocation.shape, ylocation.shape)
    positions = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    Theta = LED_angles(positions, real_config)
    if remove24:
        Theta = Theta[24:,:] # filter out bright field angles
    print(f"Samples shape: {samples.shape} Theta shape: {Theta.shape}")
    return samples_np, Theta, real_config

def load_waller_images(data_dir):
    N = 25
    # size = 616 // 2
    size = 256
    observations_raw = np.empty((N,size,size))
    for i in range(N):
        print(f'{data_dir}/ILED_{i:02d}.tif')
        img = cv2.imread(f'{data_dir}/ILED_{i:02d}.tif')/255.0 # read as BGR
        color = 2 # red
        if img.shape[0] != img.shape[1]:
            left = int((np.max(img[:,:,color].shape) - np.min(img[:,:,color].shape)) / 2)
            right = left + np.min(img[:,:,color].shape)
            observations_raw[i,:,:] = cv2.resize(img[:,left:right,color],
                                                 (size,size))
        else:
            # observations_raw[i,:,:] = img
            observations_raw[i,:,:] = img[:,:,color]

    size = observations_raw.shape[1]
    samples_np = observations_raw

    real_config = setup_config(samples_np[0,:,:], UCONN_CONFIG)

    width = np.sqrt(N)
    center = 2
    location = np.arange(width)
    xlocation = location - center
    ylocation = xlocation
    x,y = np.meshgrid(xlocation, ylocation, indexing='xy')
    positions = np.stack([y,x], axis=-1).reshape(-1,2)
    Theta = LED_angles(torch.from_numpy(positions), real_config)
    print(f"Samples shape: {samples_np.shape} Theta shape: {Theta.shape}")
    order = spiral((center,center),5)
    Theta = rearrange(Theta, order) * -1.0
    samples_np = rearrange(samples_np, order)
    print(positions)
    print(rearrange(positions, order))
    print(Theta)
    # import pdb;pdb.set_trace()
    # samples_np = rearrange(samples_np, order)
    if True:
        for slide in range(int(np.ceil(samples_np.shape[0]/40))):
            for i in range(40):
                if i+slide*40 > samples_np.shape[0]-1:
                    break
                plt.subplot(5,8,i+1)
                plt.title('raw image '+str(order[i+slide*40]))
                plt.imshow(samples_np[i+slide*40,:,:],cmap='gray')
                plt.axis('off')
            plt.show()
    return samples_np, Theta, real_config

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
    if data_source == 'umd_pi':
        samples_np, Theta, real_config = load_umd_pi_images(data_dir)
    elif data_source == 'umd_bench':
        samples_np, Theta, real_config = load_umd_bench_images(data_dir)
    elif data_source == 'matlab':
        samples_np, Theta, real_config = load_matlab_images(data_dir)
    elif data_source == 'waller':
        samples_np, Theta, real_config = load_waller_images(data_dir)

    # # visualize a few real observations
    # img1 = 'Measurement 0', samples[0,:,:]
    # img2 = 'Measurement 100', samples[100,:,:]
    # img3 = 'Measurement 200', samples[200,:,:]
    # debug_print(img1,img2,img3)

    if cj:
        # luh = np.r_[0, 9:24, 49:80, 121:168] # skip one layer for less overlap
        # luh = np.r_[0, 25:48, 121:168] # skip 2 layers for even less
        # luh = [0, 1, 5, 10, 18, 27, 39, 52, 68, 85, 105, 126, 150, 175, 203] # only horizontal
        # luh = [0, 1, 5, 10, 18]
        # print(Theta[luh])
        pred = matlab_reconstruction.real(samples_np,
                                          Theta,
                                          num_epochs=epochs,
                                          config=real_config)
        print(f'Samples shape {samples_np[0,:,:].shape}, Result shape {pred.shape}')
        # img1 = 'First Measurement', samples_np[0,:,:]
        img1 = 'Averaged', np.sum(samples_np, axis=0) / samples_np.shape[0]
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

