import numpy as np
from .img_loaders import load_image

def load_batch(paths, target_size = (224 ,224)):
    images = [load_image(path, target_size) for path in paths]
    return np.stack(images, axis=0)
