from PIL import Image
import numpy as np

def load_image(path, target_size = (224,224)):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size)

    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0

    return img_array
