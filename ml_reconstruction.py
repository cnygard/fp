import torch
import numpy as np
from forward_model import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def sim(gt_sample, Theta, num_epochs, config):
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

def real(observations, Theta, num_epochs, config):
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
