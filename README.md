# Cell Segmentation
Cell Segmentation of a human cheek cell using Laplacian of Gaussian edge detection, morphological operators, and skimage's segmentation library.
## Starting Image (Human Cheek Cell)
![](images/human_cheek.jpg)<br>
## Laplacian of Gaussian Edge Detection
```
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
```
![](images/LoG_sigma_range.jpg)<br>
A sigma value of 2 is found to reduce all outside whilst keeping edges mostly intact


