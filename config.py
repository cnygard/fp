import torch
import numpy as np

import argparse

km = 1e3
m = 1
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

parser = argparse.ArgumentParser()
parser.add_argument("name",
                    help="select name for data directory")
parser.add_argument("-s", "--source", default="umd_bench",
                    choices=["umd_bench","umd_pi","matlab","waller"], 
                    help="select camera source that data was taken from (default: umd_bench)")

parser.add_argument("--sim", action="store_true",
                    help="run in simulation mode")

parser.add_argument("-e", "--epochs", type=int, default=10,
                    help="select number of epochs to run momentum")
parser.add_argument("-lr", "--learning", type=int, default=0.01,
                    help="select learning rate")

parser.add_argument("-w", "--wavelength", default="green",
                    choices=["red","green","blue"], 
                    help="select wavelength")

parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("-d", "--debug", action="store_true",
                    help="run in debug mode")
parser.add_argument("-o", "--observations", action="store_true",
                    help="show observations")

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
    "wavelength": 532 * nm, # green wavelength
}

UMD_BENCH_CONFIG = {
    "title": "UMD Bench Microscopy",
    # "dx": 1.845e-6, # pixel size, maybe wrong see upsampling ratio in Matlab (4x)
    "dx": 1.845 * um,
    "NA_obj": 0.1,
    "NA_illum": 0.39715,
    # "LED_spacing": 4 * mm,  # distance between LEDs
    "LED_spacing": 4,  # Use this if calculating refraction
    "object_distance": 90.88 * mm,
    "pupil_radius": 350,
    "coded": True,
    "bar_angle": False,
    "Ms": [None,None,None,None,None,None,None,None],
    "wavelength": 532 * nm, # green wavelength
}

UCONN_CONFIG = {
    "title": "UConn Microscopy",
    # "dx": 1.845e-6, # pixel size, maybe wrong see upsampling ratio in Matlab (4x)
    "dx": 1.845 * um,
    # "NA_obj": 0.1,
    "NA_obj": 0.05,
    "NA_illum": 0.39715,
    # "LED_spacing": 4 * mm,  # distance between LEDs
    "LED_spacing": 4,  # Use this if calculating refraction
    "object_distance": 90.88 * mm,
    "wavelength": 532 * nm, # green wavelength
}

WALLER_CONFIG = {
    "title": "Waller Microscopy",
    "dx": 6.5 * um,
    "NA_obj": 0.1,
    "LED_dist": 4 * mm,
    "object_distance": 67.5 * mm, 
    "wavelength": 532 * nm, # green wavelength
}

# Long-range imaging configuration
FPI_CONFIG = {
    "title": "Long-range imaging",
    "dx": 10 * cm,
    "aperture_diameter": 50 * cm,
    "object_distance": 515 * km,  # Hubble Space Telescope distance
    "NA_illum": 1.5e-6,
    "wavelength": 532 * nm, # green wavelength
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
    print(
        f"Max scan distance (m) {config['object_distance'] * np.tan(np.arcsin(config['NA_illum'])):.2e}"
    )

    config["N"] = sample.shape[0]  # assume square image
    print("Object width", config["dx"] * config["N"])
    config["wavelength"] = 630 * nm  # red wavelength
    config["wavelength"] = 532 * nm  # green wavelength
    config["wavelength"] = 475 * nm  # blue wavelength
    config["k"] = 2 * np.pi / config["wavelength"]  # wave number

    config["upsample"] = 4 # ratio to scale output image
    # config["upsample"] = 2 # ratio to scale output image
    config["out_dx"] = config["upsample"] / config["dx"] # pixel size in output
    config["out_N"] = config["N"] * config["upsample"] # width of output image

    # === Define pupil mask (circular aperture) ===
    # NA_obj = config["aperture_diameter"] / (2 * config["object_distance"])
    NA_obj = config['NA_obj'] # CJCJCJ
    print(f"Objective NA {NA_obj:.2e}")
    if 'pupil_radius' in config:
        pupil_radius = config['pupil_radius']
    else:
        pupil_radius = int(NA_obj * config["N"] * config["dx"] / config["wavelength"])

    print(f"Aperture radius (pixels): {pupil_radius}")

    x = torch.arange(-config["N"] // 2, config["N"] // 2)
    Y, X = torch.meshgrid(x, x, indexing="xy")
    R = torch.sqrt(X**2 + Y**2)
    if 'coded' in config and config['coded'] == True:
        if 'bar_angle' in config:
            config["Ms"][0] = rotating_bar_mask((config['N'], config['N']), -90, 20)
            config["Ms"][1] = rotating_bar_mask((config['N'], config['N']), -67.5, 20)
            config["Ms"][2] = rotating_bar_mask((config['N'], config['N']), -27, 20)
            config["Ms"][3] = rotating_bar_mask((config['N'], config['N']), -17.5, 20)
            config["Ms"][4] = rotating_bar_mask((config['N'], config['N']), 1, 20)
            config["Ms"][5] = rotating_bar_mask((config['N'], config['N']), 12.5, 20)
            config["Ms"][6] = rotating_bar_mask((config['N'], config['N']), 30, 20)
            config["Ms"][7] = rotating_bar_mask((config['N'], config['N']), 67.5, 20)
        # config["M"] = R < 55  # circular aperture mask
        config["M"] = np.abs(Y) < pupil_radius # bar aperture
    else:
        config["M"] = R < pupil_radius  # circular aperture mask
    config["X_m"] = X * config["dx"]
    config["Y_m"] = Y * config["dx"]

    return config

#
# def setup_config_real(sample: torch.Tensor, base_config: dict):
#     """Sets up experiment parameters
#     Args:
#         sample: The sample image
#         base_config: Base optical configuration
#     """
#     config = base_config
#     # print(
#     #     f"Max scan distance (m) {config['object_distance'] * np.tan(np.arcsin(config['NA_illum'])):.2e}"
#     # )
#
#     config["N"] = sample.shape[0]  # assume square image
#     print("Object width", config["dx"] * config["N"])
#     config["wavelength"] = 630 * nm  # wavelength
#     config["k"] = 2 * np.pi / config["wavelength"]  # wave number
#     print(f'k {config['k']}')
#
#     config["upsample"] = 4 # ratio to scale output image
#     config["out_dx"] = config["upsample"] / config["dx"] # pixel size in output
#     config["out_N"] = config["N"] * config["upsample"] # width of output image
#
#     # === Define pupil mask (circular aperture) ===
#     NA_obj = config["NA_obj"]
#     print(f"Objective NA {NA_obj:.2e}")
#     pupil_radius = int(NA_obj * config["N"] * config["dx"] / config["wavelength"])
#     print(f"Aperture radius (pixels): {pupil_radius}")
#
#     x = torch.arange(-config["N"] // 2, config["N"] // 2)
#     Y, X = torch.meshgrid(x, x, indexing="xy")
#     R = torch.sqrt(X**2 + Y**2)
#     config["M"] = R < pupil_radius  # circular aperture mask
#     config["X_m"] = X * config["dx"]
#     config["Y_m"] = Y * config["dx"]
#
#     return config
