import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import rescale
import warnings
warnings.filterwarnings("ignore")

original_image = data.chelsea()

def show_images(before, after, op):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(before, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(after, cmap='gray')
    ax[1].set_title(op + " image")
    if op == "Rescaled":
        ax[0].set_xlim(0, 400)
        ax[0].set_ylim(300, 0)
    else:        
        ax[0].axis('off')
        ax[1].axis('off')
    plt.tight_layout()
        
# rescale image to 25% of the initial size
image_rescaled = rescale(original_image, 1.0 / 4.0)

show_images(original_image, image_rescaled, "Rescaled")