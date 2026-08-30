import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

km = 1e3
m = 1
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

# input variadic tuples of name,image
def debug_print(*imgs, save: bool=False):
    max_width = 3
    num_imgs = len(imgs)
    num_rows = int(np.ceil(num_imgs / max_width))
    num_cols = min(num_imgs, max_width)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.set_size_inches(16, 8, forward=True)
    for i in range(num_imgs):
        col = i % max_width
        row = i // max_width
        name,img = imgs[i]
        if num_rows > 1:
            axs[row,col].imshow(img, cmap='gray')
            axs[row,col].set_title(name)
            axs[row,col].axis('off')
        else:
            axs[col].imshow(img, cmap='gray')
            axs[col].set_title(name)
            axs[col].axis('off')
        if save:
            cv2.imwrite(f'../fpm_debug/{name}.jpg',
                        img * 255)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.show()

def matlab_round(x):
    if x >= 0:
        return int(np.floor(x + 0.5))
    else:
        return int(np.ceil(x - 0.5))

def F(x):
    return np.fft.fftshift(np.fft.fft2(x))

def iF(x):
    return np.fft.ifft2(np.fft.ifftshift(x))

def LED_angles(positions: torch.Tensor, config: dict):
    # TODO: these should be parameters in the config
    H = 90.88
    h = 1
    n = 1.52
    # startX, startY = 20, 18
    startX, startY = 0, 0
    angles = positions * config['LED_spacing']
    for led in range(positions.shape[0]):
        y0, x0 = angles[led, 0], angles[led, 1] # TODO: matlab has x and y switched. Should keep that?

        # === Calucate refraction through slide === (change LED_spacing in config to not have mm)
        l = torch.sqrt(x0**2+y0**2) # distance of LED from origin
        xoff = 0 # initial guess where beam enters bottom of slide
        thetag = -torch.asin(l/torch.sqrt(l**2+H**2)/n) # get angle of beam in glass from Snell's law
        xint = h*torch.tan(thetag) # find where the beam exits the top of the slide
        xoff = xoff-xint # modify guess where beam enters bottom of slide by this amount

        # repeat the above procedure until the beam exits the top of the slide
        # within 1 micron of center
        while torch.abs(xint) > .001:
            thetag = -torch.asin((l-xoff)/torch.sqrt((l-xoff)**2+H**2)/n)
            xint = xoff+h*torch.tan(thetag)
            xoff = xoff-xint

        # angle under the glass and angle over the cover slip
        # FPM treats this as the angle in the sample so assumes the sample has
        # refractive index = 1.0
        theta = torch.asin((l-xoff)/torch.sqrt((l-xoff)**2+H**2))
        NAt = torch.abs(torch.sin(theta));

        thetal = torch.atan2(y0, x0)
        angles[led,0] = -NAt*torch.cos(thetal)
        angles[led,1] = -NAt*torch.sin(thetal)

        # # === Angle without refraction ===
        # tanX = x0 / config['object_distance']
        # tanY = y0 / config['object_distance']
        # tans = torch.tensor([tanX, tanY])
        # atans = torch.atan(tans)
        # angles[led] = atans
        # import pdb; pdb.set_trace()
    return angles

def spiral(center,size):
    order = np.empty(size**2)
    row,col = center
    dirBorder = [
        col+1, # right
        row+1, # down
        col-1, # left
        row-1  # up
    ]
    dirIdx = 0
    i = 0
    while col < size and col >= 0 and row < size and row >= 0:
        # print(f'({row},{col})')
        order[i] = col + row*size
        i += 1
        if dirIdx == 0:
            col += 1
            d = col
        elif dirIdx == 1:
            row += 1
            d = row
        elif dirIdx == 2:
            col -= 1
            d = col
        elif dirIdx == 3:
            row -= 1
            d = row
        if d == dirBorder[dirIdx]:
            if dirIdx > 1:
                dirBorder[dirIdx] -= 1
            else:
                dirBorder[dirIdx] += 1
            dirIdx = (dirIdx + 1) % 4
    return order

def rearrange(original, order):
    if len(original) < len(order):
        print('Bad parameters >:(')
        exit()
    output = np.empty_like(original)
    for i in range(len(original)):
        # print(int(order[i]))
        output[i] = original[int(order[i])]
    # for i in range(16):
    #     for j in range(16):
    #         print(f'{j+i*16:<5}',end='')
    #     print('')
    # import pdb;pdb.set_trace()
    return output

def find_bounds(x,y,dx,dy,config):
    center_of_output = (config['out_N'] + 1) // 2
    centerX = matlab_round(center_of_output - x / dx)
    centerY = matlab_round(center_of_output - y / dy)
    lowX  = matlab_round(centerX - (config['N']-1)/2)
    lowY  = matlab_round(centerY - (config['N']-1)/2)
    highX = matlab_round(centerX + (config['N']-1)/2) + 1
    highY = matlab_round(centerY + (config['N']-1)/2) + 1
    return lowX, lowY, highX, highY

def kspace_parameterize(Theta, config):
    kx = config['k'] * Theta[:,0]
    ky = config['k'] * Theta[:,1]
    dkx = 2*np.pi/(config['dx']*config['N'])
    dky = 2*np.pi/(config['dx']*config['N'])
    return kx,ky,dkx,dky

def focal_plane_dist_to_pixels(dist_mm, config): 
    return dist_mm*mm * config["N"] * config["pixel_size"] / (config["wavelength"] * config["f1"]) # CJCJCJ fix wavelength thing

