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
from config import args

def load_umd_pi_images(data_dir):
    N = 256
    size = args.resolution
    observations_raw = np.empty((N,size,size))
    left,right,top,bottom = 225,475,175,425
    for i in range(N):
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
    return samples_np, Theta, real_config

def load_umd_bench_images(data_dir):
    N = args.samples_count
    size = args.resolution
    observations_raw = np.empty((N,size,size))
    # if args.exposures:
    #     with open(f'{data_dir}/exposures.txt', 'r') as file:
    #         exposures = [float(next(file).strip()) for _ in range(N)]
    for i in range(N):
        # print(f'{data_dir}/{i+1:04d}.jpg')
        img = cv2.imread(f'{data_dir}/{i+1:04d}.jpg', cv2.IMREAD_GRAYSCALE)/255.0
        # img = cv2.imread(f'{data_dir}/{i+1:04d}.tif', cv2.IMREAD_GRAYSCALE)/255.0
        if img.shape[0] != img.shape[1]:
            left = int((np.max(img.shape) - np.min(img.shape)) / 2)
            right = left + np.min(img.shape)
            resized = cv2.resize(img[:,left:right], (size,size))
            # exposure_adjusted = resized / (exposures[i] / np.min(exposures))
            amplitude = np.sqrt(resized)
            observations_raw[i,:,:] = amplitude
        else:
            observations_raw[i,:,:] = img

    size = observations_raw.shape[1]
    samples_np = observations_raw

    real_config = setup_config(samples_np[0,:,:], UMD_SLIT_CONFIG)

    if args.mode == "translate":
        width = N
        center = width // 2
        location = np.arange(width)
        xlocation = location - center
        ylocation = np.zeros_like(xlocation)
        positions = np.stack([xlocation,ylocation], axis=-1).reshape(-1,2)
        shift = positions * real_config['translation']
        # print(f'positions {shift}')
    elif args.mode == "rotate":
        with open(f'{data_dir}/angles.txt', 'r') as f:
            angles = [float(line.strip()) for line in f if line.strip()]
        # angles are in tuple first element, second present for consistency
        shift = np.array([[a, 0] for a in angles])
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

    real_config = setup_config(samples[0,:,:], UCONN_CONFIG)

    # === Load LED locations ===
    LED_dir = 'Data/'+args.name+'_LED_locations'
    LED_raw = scipy.io.loadmat(LED_dir)
    print(LED_raw.keys())
    xlocation, ylocation = (torch.from_numpy(LED_raw['xlocation_s'].astype(float)),
                            torch.from_numpy(LED_raw['ylocation_s'].astype(float)))
    print(xlocation.shape, ylocation.shape)
    positions = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    Theta = LED_angles(positions, real_config)
    print(f"Samples shape: {samples.shape} Theta shape: {Theta.shape}")
    return samples_np, Theta, real_config

def load_waller_images(data_dir):
    N = 25
    size = args.resolution
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
    return samples_np, Theta, real_config
