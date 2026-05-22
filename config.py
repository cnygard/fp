import torch
import numpy as np

import argparse

from util import *

parser = argparse.ArgumentParser()
parser.add_argument("data_name",
                    help="location of data relative to Data directory")
parser.add_argument("-s", "--source", default="umd_bench",
                    choices=["umd_bench","umd_pi","matlab","waller"], 
                    help="select camera source that data was taken from (default: umd_bench)")

parser.add_argument("--sim", action="store_true",
                    help="run in simulation mode")

parser.add_argument("-e", "--epochs", type=int, default=10,
                    help="select number of epochs to run momentum")
parser.add_argument("-lr", "--learning", type=int, default=0.01,
                    help="learning rate")

parser.add_argument("-wl", "--wavelength",
                    choices=["red","green","blue"])

parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-o", "--observations", action="store_true",
                    help="show observations")

parser.add_argument("-m", "--mode", type=str, default="translate",
                    choices=["translate","rotate"])
parser.add_argument("-w", "--slit_width", type=float, required=True,
                    help="slit width in mm")
parser.add_argument("-t", "--translation", type=float, required=True,
                    help="translation between measurements in mm")
parser.add_argument("-n", "--samples_count", type=int, required=True,
                    help="number of samples taken")

parser.add_argument("--save", action="store_true",
                    help="save output amplitude and phase")


args = parser.parse_args()

# # Microscopy configuration
# FPM_CONFIG = {
#     "title": "Microscopy",
#     "dx": 1.12 * um,
#     # "NA_obj": 0.1,
#     "NA_obj": 0.04986,
#     "NA_illum": 0.04986 * 4,
#     "LED_spacing": 3.45 * mm,  # Use this if calculating refraction
#     "aperture_diameter": 5.76 * mm,
#     "object_distance": 57.71 * mm,
# }

# Microscopy configuration
FPM_CONFIG = {
    "title": "UMD Pi Microscopy",
    "dx": 0.6611 * um,
    "NA_obj": 0.15,
    "NA_illum": 0.4,
    "LED_spacing": 3.45,  # Use this if calculating refraction
    "aperture_diameter": 5.76 * mm,
    "object_distance": 57.71 * mm,
}

UMD_BENCH_CONFIG = {
    "title": "UMD Bench Microscopy",
    # "dx": 1.845e-6, # pixel size, maybe wrong see upsampling ratio in Matlab (4x)
    "dx": 1.845 * um,
    "NA_obj": 0.1,
    "NA_illum": 0.39715,
    # "LED_spacing": 4 * mm,  # distance between LEDs
    "LED_spacing": 4,  # Use this if calculating refraction
    "object_distance": 450 * mm,
    "aperture_width": 350,
    "coded": True,
    "bar_angle": False,
    "Ms": [None,None,None,None,None,None,None,None],
}

UMD_SLIT_TRANS_CONFIG = {
    "title": "UMD Slit Translation Microscopy",
    # "dx": 1.845e-6, # pixel size, maybe wrong see upsampling ratio in Matlab (4x)
    "dx": 3.45 * um, # distance between pixels on the sample
    "pixel_size": 3.45 * um, # size of sensor pixel
    "object_distance": 500 * mm,
    # "aperture_width": 1.2*mm * 1080 * 3.45*um / (532*nm * 100*mm), # ~84 px
    "wavelength": 532 * nm, # green wavelength
    "f1": 125*mm, # focal length 1
    "f2": 125*mm, # focal length 2

}


UCONN_CONFIG = {
    "title": "UConn Microscopy",
    # "dx": 1.845e-6, # pixel size, maybe wrong see upsampling ratio in Matlab (4x)
    "dx": 1.845 * um,
    "NA_obj": 0.1,
    "NA_illum": 0.39715,
    # "LED_spacing": 4 * mm,  # distance between LEDs
    "LED_spacing": 4,  # Use this if calculating refraction
    "object_distance": 90.88 * mm,
    "Ms": [None,None,None,None,None,None,None,None],
}

WALLER_CONFIG = {
    "title": "Waller Microscopy",
    "dx": 6.5 * um,
    "NA_obj": 0.1,
    "LED_dist": 4 * mm,
    "object_distance": 67.5 * mm, 
}

# Long-range imaging configuration
FPI_CONFIG = {
    "title": "Long-range imaging",
    "dx": 10 * cm,
    "aperture_diameter": 50 * cm,
    "object_distance": 515 * km,  # Hubble Space Telescope distance
    "NA_illum": 1.5e-6,
}

def rotating_bar_mask(shape, angle_deg, width):
    h, w = shape
    cy, cx = h // 2, w // 2

    # coordinate grid centered at (0,0)
    y, x = np.ogrid[:h, :w]
    y = y - cy
    x = x - cx

    # convert angle to radians
    theta = np.deg2rad(angle_deg)

    # rotate coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)

    # create bar mask (threshold on rotated x axis)
    mask = np.abs(x_rot) < (width / 2)

    return mask.astype(np.uint8)

def setup_config(sample: torch.Tensor, base_config: dict):
    """Sets up experiment parameters
    Args:
        sample: The sample image
        base_config: Base optical configuration
    """
    config = base_config
    # print(
    #     f"Max scan distance (m) {config['object_distance'] * np.tan(np.arcsin(config['NA_illum'])):.2e}"
    # )

    config['N'] = sample.shape[0]  # assume square image
    print('Object width', config['dx'] * config['N'])
    if args.wavelength == 'red':
        config['wavelength'] = 630 * nm  # red wavelength
    elif args.wavelength == 'green':
        config['wavelength'] = 532 * nm  # green wavelength
    elif args.wavelength == 'blue':
        config['wavelength'] = 475 * nm  # blue wavelength
    elif args.wavelength == None and not 'wavelength' in config:
        config['wavelength'] = 532 * nm  # green wavelength
    print(f'Wavelength {args.wavelength} {config['wavelength']}')
    config['k'] = 2 * np.pi / config['wavelength']  # wave number

    config['upsample'] = 4 # ratio to scale output image
    config['out_dx'] = config['upsample'] / config['dx'] # pixel size in output
    config['out_N'] = config['N'] * config['upsample'] # width of output image

    # === Define pupil mask (circular aperture) ===
    # print(f'Objective NA {config['NA_obj']:.2e}')
    config['aperture_width'] = focal_plane_dist_to_pixels(args.slit_width, config) # pixel calculation
    config['translation'] = focal_plane_dist_to_pixels(args.translation,config)
    print(f'Slit width: {args.slit_width:.2f}mm {config['aperture_width']:.2f} pixels')
    print(f'Aperture translation: {args.translation:.2f}mm {config['translation']:.2f} pixels')
    overlap_mm = (args.slit_width - args.translation)
    overlap_px = config['aperture_width'] - config['translation']
    overlap_percent = 100 * (args.slit_width - args.translation) / args.slit_width 
    print(f'Overlap: {overlap_mm:.2f}mm {overlap_px:.2f} pixels {overlap_percent:.2f}%')

    x = torch.arange(-config['N'] // 2, config['N'] // 2)
    X, _Y = torch.meshgrid(x, x, indexing='xy')
    config['M'] = np.abs(X) < config['aperture_width']/2 # bar aperture

    return config
