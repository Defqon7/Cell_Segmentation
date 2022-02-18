import cv2
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from skimage import segmentation


def any_neighbor_zero(img, i, j):
    for k in range(-1, 2):
        for l in range(-1, 2):
            if img[i + k, j + k] == 0:
                return True
    return False


def zero_crossing(img):
    img[img > 0] = 1
    img[img < 0] = 0
    out_img = np.zeros(img.shape)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] > 0 and any_neighbor_zero(img, i, j):
                out_img[i, j] = 255
    return out_img


# read in as color so segmented border can be overlayed onto original color img
img_original = cv2.imread('human_cheek.jpg', 1)
img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

fig = plt.figure(figsize=(5, 5))
plt.gray()

# display LoG edge detection with sigma values of 1-4
for sigma in range(1, 5):
    plt.subplot(2, 2, sigma)
    result = ndimage.gaussian_laplace(img, sigma=sigma)
    plt.imshow(zero_crossing(result))
    plt.axis('off')
    plt.title('Laplacian of Gaussian with sigma=' + str(sigma), size=5)
plt.tight_layout()
plt.savefig('LoG_sigma_range.jpg')
plt.show()

# sigma of 2 removed noise whilst keeping edges mostly intact
img_edge = ndimage.gaussian_laplace(img, sigma=2)
cv2.imwrite('LoG_sigma2.jpg', img_edge)

# dilate img to fill gaps in outside edge
kernel = np.ones((7, 7), np.uint8)
dilation = cv2.dilate(img_edge, kernel, iterations=2)
cv2.imshow('Dilated', dilation)
cv2.waitKey(0)
cv2.imwrite('dilated_cell.jpg', dilation)

# erode img to original size
erosion = cv2.erode(dilation, kernel, iterations=2)
cv2.imshow('Eroded', erosion)
cv2.waitKey(0)
cv2.imwrite('eroded_cell.jpg', erosion)

# convert any pixel value greater than 1 to 255 (white)
(thresh, segmented_cell) = cv2.threshold(erosion, 1, 255, cv2.THRESH_BINARY)
cv2.imshow('Segmented Image', segmented_cell)
cv2.waitKey(0)
cv2.imwrite('segmented_cell.jpg', segmented_cell)

# overlay boundary onto original image
segmentation_boundary = segmentation.mark_boundaries(img_original, segmented_cell, mode='thick', color=(0, 0, 255))
cv2.imshow('Result', segmentation_boundary)
cv2.waitKey(0)
cv2.imwrite('segm_boundary_cell.jpg', segmentation_boundary)
cv2.destroyAllWindows()
