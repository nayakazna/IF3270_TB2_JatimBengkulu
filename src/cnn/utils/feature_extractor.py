import numpy as np
from tqdm import tqdm
import os

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from .batch_loader import load_batch


class FeatureExtractor:
    def __init__(self):
        self.model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

    def extract(self, image_paths, batch_size=32):
        features = []

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]

            batch = load_batch(batch_paths) * 255.0
            batch = preprocess_input(batch)

            feat = self.model.predict(batch, verbose=0)
            features.append(feat)

        return np.vstack(features)

    def save(self, features, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features)

    def load(self, path):
        return np.load(path)
