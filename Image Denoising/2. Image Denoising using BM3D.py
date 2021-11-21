
"""
bm3d library is not well documented yet, but looking into source code....
sigma_psd - noise standard deviation
stage_arg: Determines whether to perform hard-thresholding or Wiener filtering.
stage_arg = BM3DStages.HARD_THRESHOLDING or BM3DStages.ALL_STAGES (slow but powerful)
All stages performs both hard thresholding and Wiener filtering. 
"""

import bm3d
import cv2
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio

noisy_img = img_as_float(io.imread("./1.jpeg", as_gray=True))

BM3D_den_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

cv2.imshow("Original", noisy_img)
cv2.imshow("Denoised", BM3D_den_image)
cv2.waitKey(0)
cv2.destroyAllWindows()