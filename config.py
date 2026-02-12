import torch
import numpy as np

km = 1e3
m = 1
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9

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
    "title": "Waller Claim Microscopy",
    "dx": 870 * nm,
    "NA_obj": 0.15,
    "NA_illum": 0.4,
    "LED_spacing": 3.45,  # Use this if calculating refraction
    "aperture_diameter": 5.76 * mm,
    "object_distance": 57.71 * mm,
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
    print(
        f"Max scan distance (m) {config['object_distance'] * np.tan(np.arcsin(config['NA_illum'])):.2e}"
    )

    config["N"] = sample.shape[0]  # assume square image
    print("Object width", config["dx"] * config["N"])
    # config["wavelength"] = 475 * nm  # blue wavelength
    # config["wavelength"] = 532 * nm  # green wavelength
    config["wavelength"] = 630 * nm  # red wavelength
    config["k"] = 2 * np.pi / config["wavelength"]  # wave number

    config["upsample"] = 4 # ratio to scale output image
    config["out_dx"] = config["upsample"] / config["dx"] # pixel size in output
    config["out_N"] = config["N"] * config["upsample"] # width of output image

    # === Define pupil mask (circular aperture) ===
    # NA_obj = config["aperture_diameter"] / (2 * config["object_distance"])
    NA_obj = config['NA_obj'] # CJCJCJ
    print(f"Objective NA {NA_obj:.2e}")
    pupil_radius = int(NA_obj * config["N"] * config["dx"] / config["wavelength"])
    print(f"Aperture radius (pixels): {pupil_radius}")

    x = torch.arange(-config["N"] // 2, config["N"] // 2)
    Y, X = torch.meshgrid(x, x, indexing="xy")
    R = torch.sqrt(X**2 + Y**2)
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
