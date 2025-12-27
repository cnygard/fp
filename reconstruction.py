import torch
import numpy as np
import scipy
from forward_model import *
from util import *
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from config import *

isSim = False
epochs = 25
# data_name = 'MouseKidney_green'
data_name = 'USAF_red'
remove24 = False # remove bright field images
showObservations = False

def iterative_reconstruction_sim(gt_sample, Theta, num_epochs, config):
    # Init guesses
    dc = forward_model(gt_sample, torch.tensor([0, 0]), config)
    sample = torch.nn.Parameter(dc)
    optimizer = torch.optim.Adam([sample], lr=1e-2)

    prog_bar = tqdm(range(num_epochs))
    for _ in prog_bar:
        for theta in Theta:
            image = forward_model(sample, theta, config)
            target = forward_model(gt_sample, theta, config)

            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(sample.detach().numpy(), cmap='gray')
            # axs[0].set_title("sample (image before forward)")
            # axs[1].imshow(image.detach().numpy(), cmap='gray')
            # axs[1].set_title("Image")
            # axs[2].imshow(target.detach().numpy(), cmap='gray')
            # axs[2].set_title("Target")
            # plt.show()

            loss = torch.nn.functional.mse_loss(image, target)
            loss.backward()
            optimizer.step()
            error = torch.nn.functional.mse_loss(sample, gt_sample)
            prog_bar.set_description(f"Error: {error.item():.4f}")
            optimizer.zero_grad()
    return sample

def iterative_reconstruction_real(observations, Theta, num_epochs, config):
    # Init guesses
    N = observations.shape[0]
    num_angles = Theta.shape[0]
    print(f"Number of observations: {N}, Number of angles: {num_angles}")
    dc = np.copy(observations[0,:,:])
    # dc_amplitude = torch.tensor(np.sqrt(dc), dtype=torch.float32) 
    # sample = torch.nn.Parameter(dc_amplitude.to(torch.complex64))
    sample = torch.nn.Parameter(torch.tensor(dc)) # acculumated estimate from no angle
    optimizer = torch.optim.Adam([sample], lr=1e-3)

    # need angle per measurement to reconstruct
    # call forward model on sample, image = forward(sample, angle of current observation)
    # TODO: find observation angles, LED_location in Matlab -> save to .mat -> load here?
    # TODO: ^ or find a new dataset with known angles?
    prog_bar = tqdm(range(num_epochs))
    for _ in prog_bar:
        for i in range(N):
            # image = sample # reproject current estimate to same angle as this observation
            image = forward_model(sample, Theta[i, :], config, observations[i,:,:])
            target = observations[i,:,:]

            # ISSUES is in forward_model, see visualizations from there, compare sim vs real?
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(sample.detach().numpy(), cmap='gray')
            # axs[0].set_title("sample (image before forward)")
            # axs[1].imshow(image.detach().numpy(), cmap='gray')
            # axs[1].set_title("Image")
            # axs[2].imshow(target.detach().numpy(), cmap='gray')
            # axs[2].set_title("Target")
            # plt.show()

            loss = torch.nn.functional.mse_loss(image, target)
            # print(f"Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            # error = torch.nn.functional.mse_loss(sample, gt_sample)
            # prog_bar.set_description(f"Error: {error.item():.4f}")
            optimizer.zero_grad()
    return sample

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
    sample = sample / 255.0  # normalize
    sample = torch.from_numpy(sample)

    sim_config = setup_config(sample, FPM_CONFIG)

    theta_max = np.rad2deg(np.arcsin(config["NA_illum"]))
    theta_max = np.rad2deg(np.arcsin(sim_config["NA_illum"]))
    print(f"Max rotation (deg) {theta_max:.2e}")
    theta = torch.linspace(-theta_max, theta_max, 10)
    theta_y, theta_x = torch.meshgrid(theta, theta, indexing="ij")
    Theta = torch.stack([theta_y, theta_x], dim=-1).reshape(-1, 2)
    print(f"Theta: {Theta}")

    pred = iterative_reconstruction_sim(sample, Theta, num_epochs=epochs, config=sim_config)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(forward_model(sample, torch.tensor([0, 0]), sim_config))
    axs[0].set_title("Measurement")
    axs[1].imshow(pred.detach().cpu())
    axs[1].set_title("FP Reconstruction")
    axs[2].imshow(sample)
    axs[2].set_title("GT")
    plt.show(block=False)
elif __name__ == "__main__" and not isSim:
    # === Load and preprocess image ===
    data_dir = 'Data/'+data_name
    LED_dir = 'Data/'+data_name+'_LED_locations'
    observations_raw = scipy.io.loadmat(data_dir)['imlow_HDR'].transpose(2,0,1) # first index is now image index

    N = observations_raw.shape[0]
    samples_np = np.zeros((N, 256, 256), dtype=np.float32)
    for i in range(N):
        samples_np[i,:,:] = cv2.resize(observations_raw[i,:,:], (256, 256)) / 255.0
    samples = torch.from_numpy(samples_np)
    if remove24:
        samples = samples[24:,:,:] # filter out bright field images

    #
    # for slide in range(int(np.ceil(samples_np.shape[2]/40))):
    #     for i in range(40):
    #         if i+slide*40 > samples_np.shape[2]-1:
    #             break
    #         plt.subplot(5,8,i+1)
    #         plt.title('raw image '+str(i+slide*40+1))
    #         plt.imshow(samples_np[i+slide*40,:,:],cmap='gray')
    #         plt.axis('off')
    #     plt.show()

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

    real_config = setup_config_real(samples[0,:,:], UCONN_CONFIG)
    # load LED locations
    LED_raw = scipy.io.loadmat(LED_dir)
    print(LED_raw.keys())
    xlocation, ylocation = torch.from_numpy(LED_raw['xlocation_s'].astype(float)), torch.from_numpy(LED_raw['ylocation_s'].astype(float))
    print(xlocation.shape, ylocation.shape)
    positions = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    # Theta = LED_angles(positions, real_config) # TODO should use this
    Theta = positions
    # print(f"Theta: {Theta}")
    Theta = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    if remove24:
        Theta = Theta[24:,:] # filter out bright field angles
    print(f"Samples shape: {samples.shape} Theta shape: {Theta.shape}")
    # print(f"Theta: {Theta}")

    # visualize a few real observations
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(samples[0,:,:], cmap='gray')
    axs[0].set_title("Measurement 0")
    axs[1].imshow(samples[100,:,:], cmap='gray')
    axs[1].set_title("Measurement 100")
    axs[2].imshow(samples[200,:,:], cmap='gray')
    axs[2].set_title("Measurement 200")
    plt.show(block=False)

    # theta_max = np.rad2deg(np.arcsin(config["NA_illum"]))
    # print(f"Max rotation (deg) {theta_max:.2e}")
    # theta = torch.linspace(-theta_max, theta_max, 10)
    # theta_y, theta_x = torch.meshgrid(theta, theta, indexing="ij")
    # Theta = torch.stack([theta_y, theta_x], dim=-1).reshape(-1, 2)

    pred = iterative_reconstruction_real(samples, Theta, num_epochs=epochs, config=real_config)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(samples[0,:,:], cmap='gray')
    axs[0].set_title("First Measurement")
    axs[1].imshow(pred.detach().cpu().numpy().astype('float'), cmap='gray')
    axs[1].set_title("FP Reconstruction")
    # axs[2].imshow(sample)
    # axs[2].set_title("GT")
    plt.show()
