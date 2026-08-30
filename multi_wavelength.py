import numpy as np

def deconstruct_measurement_wavelengths(input_image,
                                        pupil_offset_x_m,
                                        pupil_offset_y_m,
                                        config):
    ny, nx = input_image.shape
    ws = config['wavelength']
    wc = len(ws)

    F = np.fft.fftshift(np.fft.fft2(input_image))

    pupil = np.zeros_like(input_image)
    simple_separation = np.empty((ny,nx,wc))

    x = (np.arange(nx) - nx//2)
    y = (np.arange(ny) - ny//2)
    X,Y = np.meshgrid(x,y)
    for i in range(wc):
        scale = config['pixel_size'] * nx / (ws[i] * config['f1'])

        x0 = pupil_offset_x_m * scale
        y0 = pupil_offset_y_m * scale

        radius = config['aperture_diameter_m'] * scale / 2
        rho = np.sqrt((X - x0) ** 2 +
                      (Y - y0) ** 2)

        pupil_function = (rho <= radius).astype(np.float64)

        F_filtered = F * pupil_function
        img_filtered = np.fft.ifft2(np.fft.ifftshift(F_filtered))
        simple_separation[:,:,i] = np.abs(img_filtered)**2 # should abs**2?
        # accumulated = accumulated + np.abs(img_filtered)**2
        # pupil = pupil + F_filtered

    intensity_sum = np.sum(simple_separation, axis=2)

    ts = np.empty_like(simple_separation)
    Is = simple_separation
    for i in range(wc):
        t = np.zeros_like(ts[:,:,i])
        for j in range(wc):
            t += Is[:,:,j]
        ts[:,:,i] = t
        Is[:,:,i] = Is[:,:,i] * input_image / (t+1e-10)
        # for j in range(wc - (wc - i)):
        #     print(f'{j}`',end='')
        #     t += Is[:,:,j]
        # for j in range(wc - i):
        #     print(f'{j+i} ',end='')
        #     t += Is[:,:,j]
        # print('')

    import matplotlib.pyplot as plt
    from util import debug_print
    img1 = 'c1',Is[:,:,0]
    img2 = 'c2',Is[:,:,1]
    img3 = 'c3',Is[:,:,2]
    debug_print(img1,img2,img3,save=False)
    exit(1)
    return np.sqrt(Is)
