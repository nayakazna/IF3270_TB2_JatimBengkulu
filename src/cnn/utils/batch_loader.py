import numpy as np
from .img_loaders import load_image
from concurrent.futures import ThreadPoolExecutor

def load_batch(paths, target_size=(224, 224), max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        images = list(ex.map(lambda p: load_image(p, target_size), paths))
    return np.stack(images, axis=0)
