import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
F = np.fft.fftshift(np.fft.fft2(img))
c = img.shape[0] // 2
size = 50


y, x = np.ogrid[:img.shape[0], :img.shape[1]]
dist_from_center = np.sqrt((x - c+60)**2 + (y - c)**2)
mask = dist_from_center <= size
F_small = np.where(mask, F, 0)



# F_small = np.zeros_like(F)
# mid = np.s_[c-size:c+size,c-size:c+size]
# F_small[mid] = F[mid]
blur = np.fft.ifft2(np.fft.ifftshift(F_small))

fig,axs = plt.subplots(2,2)
axs[0,0].imshow(img,cmap='gray')
axs[0,1].imshow(np.real(np.log(F)),cmap='gray')
axs[1,0].imshow(np.real(blur),cmap='gray')
axs[1,1].imshow(np.real(np.log(F_small)),cmap='gray')
plt.tight_layout()
plt.show()
