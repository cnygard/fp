import torch
import numpy as np

import argparse
import tomllib

from util import *

init_parser = argparse.ArgumentParser(add_help=False)
init_parser.add_argument("data_dir", nargs='?',
                    help="location of data relative to Data directory")
init_args, remaining_args = init_parser.parse_known_args()

parser = argparse.ArgumentParser()
parser.add_argument("data_dir",
                    help="location of data to be reconstructed")
parser.add_argument("--data_name", help="select file name of data (for sim)")
parser.add_argument("-s", "--source", default="umd_bench",
                    choices=["umd_bench","umd_pi","matlab","waller","sim","array_sim"], 
                    help="select camera source that data was taken from (default: umd_bench)")

# parser.add_argument("--sim", action="store_true",
#                     help="run in simulation mode")

parser.add_argument("-e", "--epochs", type=int, default=10,
                    help="select number of epochs to run momentum")
parser.add_argument("-l", "--learning", type=float, default=0.1,
                    help="learning rate")

parser.add_argument("--wavelength", choices=["red","green","blue"]) # CJCJCJ change to allow for multiple wavelengths

parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-o", "--observations", action="store_true",
                    help="show observations")

parser.add_argument("-m", "--mode", type=str, default="tri-color",
                    choices=["tri-color", "traditional"])
parser.add_argument("-w", "--aperture_diameter", type=float,
                    help="aperture diameter in mm")
parser.add_argument("-t", "--translation", type=float,
                    help="translation between measurements in mm")
parser.add_argument("-n", "--samples_count", type=int,
                    help="number of samples taken")
parser.add_argument("-r", "--resolution", type=int, default=1080,
                    help="desired pixel width of square samples")
# parser.add_argument("--dir", default="l", choices=["l","r"], 
#                     help="direction of panning (l = left to right, r = right to left)")
# parser.add_argument("-ex", "--exposures", action="store_true",
#                     help="allows changing exposures per measurement. reads from exposures.txt in data set")
parser.add_argument("-u", "--upsample", type=int, default=2,
                    help="amount to upscale output width")
parser.add_argument("--momentum", type=int,
                    help="use momentum and specify epochs")
parser.add_argument("--file_extension", default=".tif",
                    help="specify file extention for data")

parser.add_argument("--save", action="store_true",
                    help="save output amplitude and phase")

if init_args.data_dir:
  remaining_args.append(init_args.data_dir.removesuffix('/'))
  with open(f'{init_args.data_dir}/config.toml', "rb") as f:
      toml_defaults = tomllib.load(f)
  parser.set_defaults(**toml_defaults)
args = parser.parse_args(remaining_args)

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

UMD_SLIT_CONFIG = {
    "title": "UMD Slit Translation Microscopy",
    # "dx": 1.845e-6, # pixel size, maybe wrong see upsampling ratio in Matlab (4x)
    "dx": 3.45 * um, # distance between pixels on the sample
    "pixel_size": 3.45 * um, # size of sensor pixel
    "object_distance": 500 * mm,
    # "aperture_diameter_px": 1.2*mm * 1080 * 3.45*um / (532*nm * 100*mm), # ~84 px
    "wavelength": 532 * nm, # green wavelength
    "f1": 125*mm, # focal length 1
    "f2": 125*mm, # focal length 2
}

ARRAY_CONFIG = {
    "title": "Simulated FP Camera Array",
    # "dx": 1.845e-6, # pixel size, maybe wrong see upsampling ratio in Matlab (4x)
    "dx": 148/3 * um, # distance between pixels on the sample
    "pixel_size": 1.85 * um, # size of sensor pixel
    "object_distance": 2 * m,
    # "aperture_diameter_px": 1.2*mm * 1080 * 3.45*um / (532*nm * 100*mm), # ~84 px
    "f_number": 22,
    "wavelength": [653*nm,515*nm,445*nm],
    "f1": 75*mm, # focal length 1
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

def setup_config(sample: torch.Tensor, base_config: dict):
    """Sets up experiment parameters
    Args:
        sample: The sample image
        base_config: Base optical configuration
    """
    config = base_config

    config['N'] = sample.shape[0]  # assume square image
    print(f'Object width {config['dx'] * config['N']} m')
    if args.wavelength == 'red':
        config['wavelength'] = 630 * nm  # red wavelength
    elif args.wavelength == 'green':
        config['wavelength'] = 532 * nm  # green wavelength
    elif args.wavelength == 'blue':
        config['wavelength'] = 475 * nm  # blue wavelength
    elif args.wavelength == None and not 'wavelength' in config:
        config['wavelength'] = 532 * nm  # green wavelength
    print(f'Wavelength {args.wavelength} {config['wavelength']}')
    if args.mode == 'tri-color':
        config['k'] = [1,2,3]
        for w in range(len(config['wavelength'])):
            config['k'][w] = 2 * np.pi / config['wavelength'][w]  # wave number
    else:
        config['k'] = 2 * np.pi / config['wavelength']  # wave number

    config['upsample'] = args.upsample # ratio to scale output image
    config['out_dx'] = config['upsample'] / config['dx'] # pixel size in output
    config['out_N'] = config['N'] * config['upsample'] # width of output image

    # === Define pupil mask (circular aperture) ===
    # print(f'Objective NA {config['NA_obj']:.2e}')
    config['aperture_diameter_m'] = args.aperture_diameter * mm
    config['aperture_diameter_px'] = focal_plane_dist_to_pixels(args.aperture_diameter, config) # pixel calculation

    config['translation_px'] = focal_plane_dist_to_pixels(args.translation,config)
    print(f'Slit width: {args.aperture_diameter:.2f}mm {config['aperture_diameter_px']:.2f} pixels {100*config['aperture_diameter_px']/config['out_N']:.2f}% of output')
    print(f'Aperture translation: {args.translation:.2f}mm {config['translation_px']:.2f}px')
    radius_px = config['aperture_diameter_px']/2.0
    overlap_area_px = 2*(radius_px**2)*np.arccos(config['translation_px']/(2*radius_px))-(config['translation_px']/2)*np.sqrt(4*(radius_px**2)-(config['translation_px']**2))
    overlap_percent = 100*overlap_area_px/(np.pi * (radius_px**2))
    print(f'Overlap: {overlap_area_px:.2f}mm {overlap_percent:.2f}%')
    # coverage_mm = args.aperture_diameter + (args.aperture_diameter - overlap_mm) * (args.samples_count - 1)
    # coverage_px = config['aperture_diameter_px'] + (config['aperture_diameter_px'] - overlap_px) * (args.samples_count - 1)
    # coverage_percent = 100 * coverage_px / config['out_N']
    # print(f'Coverage: {coverage_mm:.2f}mm {coverage_px:.2f}px {coverage_percent:.2f}%')

    x = torch.arange(-config['N'] // 2, config['N'] // 2)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    R = torch.sqrt(X**2 + Y**2)
    config['M'] = R < config['aperture_diameter_px']/2

    print(f'Learning rate: {args.learning}')
    return config
