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
epochs = 2
# data_name = 'MouseKidney_green'
data_name = 'USAF_red'
remove24 = False # remove bright field images
showObservations = False
cj = True # use CJ's reconstruction from Matlab code

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

def matlab_round(x):
    if x >= 0:
        return int(np.floor(x + 0.5))
    else:
        return int(np.ceil(x - 0.5))

def F(x):
    return np.fft.fftshift(np.fft.fft2(x))

def iF(x):
    return np.fft.ifft2(np.fft.ifftshift(x))

def cj_reconstruction_sim(gt_sample, Theta, num_epochs, config):
    num_angles = Theta.shape[0]
    
    print(f"Number of angles: {num_angles}")
    guess = np.zeros((gt_sample.shape[0], gt_sample.shape[1]))
    # guess = np.copy(observations[0,:,:])
    # guess = np.copy(gt_sample)
    guess = cv2.resize(guess,
                       (guess.shape[0] * config['upsample'],
                        guess.shape[1] * config['upsample']))
    guessFT = F(guess)
    kx = config['k'] * Theta[:,0]
    ky = config['k'] * Theta[:,1]
    dkx = 2*np.pi/(config['dx']*config['N'])
    dky = 2*np.pi/(config['dx']*config['N'])

    region = np.copy(guessFT)
    # region = np.zeros(guessFT.shape)

    cv2.imwrite(f'../CJsim/before.jpg',
                np.log(np.real(region)) * 255)

    # need angle per measurement to reconstruct
    # call replace-angle on sample
    prog_bar = tqdm(range(num_epochs))
    first = True
    for _ in prog_bar:
        for i in range(num_angles):
            # image = sample # reproject current estimate to same angle as this observation
            output_center = config['out_N'] // 2
            # find bounds of where in guess to replace
            r_centerX = int(np.round(output_center - kx[i].item() / dkx))
            r_centerY = int(np.round(output_center - ky[i].item() / dky))
            # print(f'X center {r_centerX} outCenter {output_center} kxi {kx[i].item()} dkx {dkx} config k {config['k']}')
            # if not first:
            #     import pdb; pdb.set_trace()
            r_lowX = r_centerX - config['N']//2
            r_lowY = r_centerY - config['N']//2
            r_highX = r_centerX + config['N']//2
            r_highY = r_centerY + config['N']//2
            r = guessFT[r_lowY:r_highY, r_lowX:r_highX]
            r_masked = r * config['M'].detach().cpu().numpy()
            low_res_guess = iF(r_masked)
            measurement = config['upsample']**2 * gt_sample
            low_res_guess = measurement * np.exp(1j*np.angle(low_res_guess)) # replace amplitude but preserve phase
            low_res_guess_FT = F(low_res_guess)
            guessFT[r_lowY:r_highY,r_lowX:r_highX] =     \
                guessFT[r_lowY:r_highY,r_lowX:r_highX] + \
                config['M'].detach().cpu().numpy() *     \
                (low_res_guess_FT - r_masked)
            if False and first:
                toBeChanged = region[r_lowY:r_highY,r_lowX:r_highX]
                toBeAdded = config['M'].detach().cpu().numpy() * \
                            (low_res_guess_FT - r_masked)
                region[r_lowY:r_highY,r_lowX:r_highX] =     \
                    region[r_lowY:r_highY,r_lowX:r_highX] + \
                    config['M'].detach().cpu().numpy() *    \
                    (low_res_guess_FT - r_masked)
                guess = iF(guessFT)
                fig, axs = plt.subplots(2, 3)
                fig.set_size_inches(16, 9, forward=True)
                axs[0,0].imshow(np.real(iF(r_masked)), cmap='gray')
                axs[0,0].set_title("Measurement")
                axs[0,1].imshow(np.real(guess), cmap='gray')
                axs[0,1].set_title("Guess")
                axs[0,2].imshow(np.log(np.real(region)), cmap='gray')
                axs[0,2].set_title(f'Guess Spectrum, center=({r_centerX},{r_centerY})')
                axs[1,0].imshow(np.log(np.real(toBeChanged))/255, cmap='gray')
                axs[1,0].set_title("To Be Changed")
                axs[1,1].imshow(np.log(np.real(toBeAdded))/255, cmap='gray')
                axs[1,1].set_title("To Be Added")
                axs[1,2].imshow(np.zeros(toBeAdded.shape), cmap='gray')
                axs[1,2].set_title("Zero")
                plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
                plt.show()
                # cv2.imwrite(f'../CJsim/spectrum_reconstruction_{i}.jpg',
                #             np.log(np.real(region)) * 255)
        first = False
    return iF(guessFT)

def cj_reconstruction_real(observations, Theta, num_epochs, config):
    num_samples = observations.shape[0]
    num_angles = Theta.shape[0]
    
    print(f"Number of observations: {num_samples}, Number of angles: {num_angles}")
    # guess = np.zeros((observations.shape[1], observations.shape[2]))
    # guess = np.copy(observations[0,:,:])
    guess = np.sum(observations, axis=0)
    guess = cv2.resize(guess,
                       (guess.shape[0] * config['upsample'],
                        guess.shape[1] * config['upsample']))
    guessFT = F(guess)
    kx = config['k'] * Theta[:,0]
    ky = config['k'] * Theta[:,1]
    dkx = 2*np.pi/(config['dx']*config['N'])
    dky = 2*np.pi/(config['dx']*config['N'])
    # print(f'kx {kx}')
    # import pdb; pdb.set_trace()

    print(f'kx {kx}')

    # region = np.copy(guessFT)
    region = np.zeros(guessFT.shape)

    cv2.imwrite(f'../CJ/before.jpg',
                np.log(np.real(region)) * 255)

    # need angle per measurement to reconstruct
    # call replace-angle on sample
    prog_bar = tqdm(range(num_epochs))
    first = True
    for _ in prog_bar:
        for i in range(num_samples):
            # image = sample # reproject current estimate to same angle as this observation
            output_center = (config['out_N']+1) / 2
            # find bounds of where in guess to replace
            r_centerX = matlab_round(output_center - kx[i].item() / dkx)
            r_centerY = matlab_round(output_center - ky[i].item() / dky)
            # print(f'X center {r_centerX} outCenter {output_center} kxi {kx[i].item()} dkx {dkx} config k {config['k']}')
            # if not first:
            #     import pdb; pdb.set_trace()
            r_lowX  = matlab_round(r_centerX - (config['N']-1)/2)
            r_lowY  = matlab_round(r_centerY - (config['N']-1)/2)
            r_highX = matlab_round(r_centerX + (config['N']-1)/2) + 1
            r_highY = matlab_round(r_centerY + (config['N']-1)/2) + 1
            # import pdb;pdb.set_trace()
            r = guessFT[r_lowY:r_highY, r_lowX:r_highX]
            if r.shape == (256,0):
                import pdb; pdb.set_trace()
            r_masked = r * config['M'].detach().cpu().numpy()
            low_res_guess = iF(r_masked)
            measurement = config['upsample']**2 * observations[i,:,:]
            low_res_guess = measurement * np.exp(1j*np.angle(low_res_guess)) # replace amplitude but preserve phase
            low_res_guess_FT = F(low_res_guess)
            guessFT[r_lowY:r_highY,r_lowX:r_highX] =     \
                guessFT[r_lowY:r_highY,r_lowX:r_highX] + \
                config['M'].detach().cpu().numpy() *     \
                (low_res_guess_FT - r_masked)
            if False and first:
                region[r_lowY:r_highY,r_lowX:r_highX] =     \
                    region[r_lowY:r_highY,r_lowX:r_highX] + \
                    config['M'].detach().cpu().numpy() *    \
                    (low_res_guess_FT - r_masked)
                guess = iF(guessFT)
                fig, axs = plt.subplots(1, 3)
                fig.set_size_inches(16, 8, forward=True)
                axs[0].imshow(observations[i,:,:], cmap='gray')
                axs[0].set_title("Measurement")
                axs[1].imshow(np.real(guess), cmap='gray')
                axs[1].set_title("Guess")
                axs[2].imshow(np.log(np.real(region)), cmap='gray')
                axs[2].set_title("Guess Spectrum")
                plt.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0)
                plt.show()
                # cv2.imwrite(f'../CJ/spectrum_reconstruction_{i}.jpg',
                #             np.log(np.real(region)) * 255)
        first = False
    return iF(guessFT)

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

    sim_config = setup_config(sample, FPM_CONFIG)

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
    print(positions)
    Theta = LED_angles(positions, sim_config)

    print(f"Theta: {Theta}")

    if cj:
        # Theta = torch.deg2rad(Theta)
        pred = cj_reconstruction_sim(sample_np, Theta,
                                      num_epochs=epochs,
                                      config=sim_config)
        print(f'Sample shape {sample.shape}, Result shape {pred.shape}')
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(16, 8, forward=True)
        axs[0].imshow(sample, cmap='gray')
        axs[0].set_title("Ground Truth")
        axs[1].imshow(np.abs(pred.astype('float')), cmap='gray')
        axs[1].set_title("Amplitude")
        axs[2].imshow(np.angle(pred.astype('float')), cmap='gray')
        axs[2].set_title("Phase")
        plt.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0)
        plt.show()
    else:
        pred = iterative_reconstruction_sim(sample, Theta, num_epochs=epochs, config=sim_config)
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(forward_model(sample, torch.tensor([0, 0]), sim_config))
        axs[0].set_title("Measurement")
        axs[1].imshow(pred.detach().cpu())
        axs[1].set_title("FP Reconstruction")
        axs[2].imshow(sample)
        axs[2].set_title("GT")
        plt.show()
elif __name__ == "__main__" and not isSim:
    # === Load and preprocess image ===
    data_dir = 'Data/'+data_name
    LED_dir = 'Data/'+data_name+'_LED_locations'
    observations_raw = scipy.io.loadmat(data_dir)['imlow_HDR'].transpose(2,0,1) # first index is now image index

    # === Setup observations ===
    N = observations_raw.shape[0]
    size = 141
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

    real_config = setup_config_real(samples[0,:,:], UCONN_CONFIG)

    # === Load LED locations ===
    LED_raw = scipy.io.loadmat(LED_dir)
    print(LED_raw.keys())
    xlocation, ylocation = torch.from_numpy(LED_raw['xlocation_s'].astype(float)), torch.from_numpy(LED_raw['ylocation_s'].astype(float))
    print(xlocation.shape, ylocation.shape)
    positions = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    print(positions)
    Theta = LED_angles(positions, real_config)
    # Theta = positions
    # xs = torch.tensor([0,-0.0437,-0.0436,-0.0000,0.0436,0.0437,0.0436,-0.0000,-0.0436,-0.0870,-0.0871,-0.0870,-0.0867,-0.0435,-0.0000,0.0435,0.0867,0.0870,0.0871,0.0870,0.0867,0.0435,-0.0000,-0.0435,-0.0867,-0.1295,-0.1299,-0.1300,-0.1299,-0.1295,-0.1289,-0.0863,-0.0433,-0.0000,0.0433,0.0863,0.1289,0.1295,0.1299,0.1300,0.1299,0.1295,0.1289,0.0863,0.0433,-0.0000,-0.0433,-0.0863,-0.1289,-0.1708,-0.1716,-0.1720,-0.1722,-0.1720,-0.1716,-0.1708,-0.1697,-0.1281,-0.0858,-0.0430,-0.0000,0.0430,0.0858,0.1281,0.1697,0.1708,0.1716,0.1720,0.1722,0.1720,0.1716,0.1708,0.1697,0.1281,0.0858,0.0430,-0.0000,-0.0430,-0.0858,-0.1281,-0.1697,-0.2104,-0.2117,-0.2127,-0.2133,-0.2135,-0.2133,-0.2127,-0.2117,-0.2104,-0.2088,-0.1684,-0.1270,-0.0851,-0.0427,-0.0000,0.0427,0.0851,0.1270,0.1684,0.2088,0.2104,0.2117,0.2127,0.2133,0.2135,0.2133,0.2127,0.2117,0.2104,0.2088,0.1684,0.1270,0.0851,0.0427,-0.0000,-0.0427,-0.0851,-0.1270,-0.1684,-0.2088,-0.2482,-0.2501,-0.2516,-0.2527,-0.2534,-0.2536,-0.2534,-0.2527,-0.2516,-0.2501,-0.2482,-0.2459,-0.2068,-0.1667,-0.1258,-0.0842,-0.0422,-0.0000,0.0422,0.0842,0.1258,0.1667,0.2068,0.2459,0.2482,0.2501,0.2516,0.2527,0.2534,0.2536,0.2534,0.2527,0.2516,0.2501,0.2482,0.2459,0.2068,0.1667,0.1258,0.0842,0.0422,-0.0000,-0.0422,-0.0842,-0.1258,-0.1667,-0.2068,-0.2459,-0.2838,-0.2864,-0.2886,-0.2903,-0.2915,-0.2923,-0.2926,-0.2923,-0.2915,-0.2903,-0.2886,-0.2864,-0.2838,-0.2808,-0.2433,-0.2046,-0.1649,-0.1244,-0.0833,-0.0418,-0.0000,0.0418,0.0833,0.1244,0.1649,0.2046,0.2433,0.2808,0.2838,0.2864,0.2886,0.2903,0.2915,0.2923,0.2926,0.2923,0.2915,0.2903,0.2886,0.2864,0.2838,0.2808,0.2433,0.2046,0.1649,0.1244,0.0833,0.0418,-0.0000,-0.0418,-0.0833,-0.1244,-0.1649,-0.2046,-0.2433,-0.2808])
    #
    # ys = torch.tensor([0,0,-0.0436,-0.0437,-0.0436,-0.0000,0.0436,0.0437,0.0436,0.0435,0,-0.0435,-0.0867,-0.0870,-0.0871,-0.0870,-0.0867,-0.0435,-0.0000,0.0435,0.0867,0.0870,0.0871,0.0870,0.0867,0.0863,0.0433,0,-0.0433,-0.0863,-0.1289,-0.1295,-0.1299,-0.1300,-0.1299,-0.1295,-0.1289,-0.0863,-0.0433,-0.0000,0.0433,0.0863,0.1289,0.1295,0.1299,0.1300,0.1299,0.1295,0.1289,0.1281,0.0858,0.0430,0,-0.0430,-0.0858,-0.1281,-0.1697,-0.1708,-0.1716,-0.1720,-0.1722,-0.1720,-0.1716,-0.1708,-0.1697,-0.1281,-0.0858,-0.0430,-0.0000,0.0430,0.0858,0.1281,0.1697,0.1708,0.1716,0.1720,0.1722,0.1720,0.1716,0.1708,0.1697,0.1684,0.1270,0.0851,0.0427,0,-0.0427,-0.0851,-0.1270,-0.1684,-0.2088,-0.2104,-0.2117,-0.2127,-0.2133,-0.2135,-0.2133,-0.2127,-0.2117,-0.2104,-0.2088,-0.1684,-0.1270,-0.0851,-0.0427,-0.0000,0.0427,0.0851,0.1270,0.1684,0.2088,0.2104,0.2117,0.2127,0.2133,0.2135,0.2133,0.2127,0.2117,0.2104,0.2088,0.2068,0.1667,0.1258,0.0842,0.0422,0,-0.0422,-0.0842,-0.1258,-0.1667,-0.2068,-0.2459,-0.2482,-0.2501,-0.2516,-0.2527,-0.2534,-0.2536,-0.2534,-0.2527,-0.2516,-0.2501,-0.2482,-0.2459,-0.2068,-0.1667,-0.1258,-0.0842,-0.0422,-0.0000,0.0422,0.0842,0.1258,0.1667,0.2068,0.2459,0.2482,0.2501,0.2516,0.2527,0.2534,0.2536,0.2534,0.2527,0.2516,0.2501,0.2482,0.2459,0.2433,0.2046,0.1649,0.1244,0.0833,0.0418,0,-0.0418,-0.0833,-0.1244,-0.1649,-0.2046,-0.2433,-0.2808,-0.2838,-0.2864,-0.2886,-0.2903,-0.2915,-0.2923,-0.2926,-0.2923,-0.2915,-0.2903,-0.2886,-0.2864,-0.2838,-0.2808,-0.2433,-0.2046,-0.1649,-0.1244,-0.0833,-0.0418,-0.0000,0.0418,0.0833,0.1244,0.1649,0.2046,0.2433,0.2808,0.2838,0.2864,0.2886,0.2903,0.2915,0.2923,0.2926,0.2923,0.2915,0.2903,0.2886,0.2864,0.2838,0.2808])
    # Theta = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
    # # print(f"Theta: {Theta}")
    # # Theta = torch.stack([ylocation, xlocation], dim=-1).reshape(-1, 2)
    # if remove24:
    #     Theta = Theta[24:,:] # filter out bright field angles
    print(f"Samples shape: {samples.shape} Theta shape: {Theta.shape}")
    #
    # # visualize a few real observations
    # # fig, axs = plt.subplots(1, 3)
    # # axs[0].imshow(samples[0,:,:], cmap='gray')
    # # axs[0].set_title("Measurement 0")
    # # axs[1].imshow(samples[100,:,:], cmap='gray')
    # # axs[1].set_title("Measurement 100")
    # # axs[2].imshow(samples[200,:,:], cmap='gray')
    # # axs[2].set_title("Measurement 200")
    # # plt.show()
    #
    # # theta_max = np.rad2deg(np.arcsin(config["NA_illum"]))
    # # print(f"Max rotation (deg) {theta_max:.2e}")
    # # theta = torch.linspace(-theta_max, theta_max, 10)
    # # theta_y, theta_x = torch.meshgrid(theta, theta, indexing="ij")
    # # Theta = torch.stack([theta_y, theta_x], dim=-1).reshape(-1, 2)
    # print(f"Theta: {Theta}")

    if cj:
        pred = cj_reconstruction_real(samples_np, Theta,
                                      num_epochs=epochs,
                                      config=real_config)
        print(f'Samples shape {samples[0,:,:].shape}, Result shape {pred.shape}')
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(16, 8, forward=True)
        axs[0].imshow(samples[0,:,:], cmap='gray')
        axs[0].set_title("First Measurement")
        axs[1].imshow(np.abs(pred.astype('float')), cmap='gray')
        axs[1].set_title("Amplitude")
        axs[2].imshow(np.angle(pred.astype('float')), cmap='gray')
        axs[2].set_title("Phase")
        plt.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0)
        plt.show()
    else:
        pred = iterative_reconstruction_real(samples, Theta, num_epochs=epochs, config=real_config)
        print(f'Samples shape {samples[0,:,:].shape}, Result shape {pred.shape}')
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(samples[0,:,:], cmap='gray')
        axs[0].set_title("First Measurement")
        axs[1].imshow(pred.detach().cpu().numpy().astype('float'), cmap='gray')
        axs[1].set_title("FP Reconstruction")
        # axs[2].imshow(sample)
        # axs[2].set_title("GT")
        plt.show()
