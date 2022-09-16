# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    image_rescaled = rescale(image, 0.25, anti_aliasing=False)
    image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
    image_downscaled = downscale_local_mean(image, (4, 3))
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax1 = axes.ravel()
    ax1[0].imshow(image, cmap='gray')
    ax1[0].set_title("Original image")
    ax1[1].imshow(image_rescaled, cmap='gray')
    ax1[1].set_title("Rescaled image (aliasing)")
    ax1[2].imshow(image_resized, cmap='gray')
    ax1[2].set_title("Resized image (no aliasing)")
    ax1[3].imshow(image_downscaled, cmap='gray')
    ax1[3].set_title("Downscaled image (no aliasing)")
    #ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #ax.set_title("Training: %i" % label)
    plt.tight_layout()
    plt.show()
