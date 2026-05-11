import os

import numpy as np
from tqdm import tqdm

from src.cnn.utils.batch_loader import load_batch


class FeatureExtractor:
    def __init__(self, model_name="inception_v3", pooling="avg"):
        self.model_name = model_name.lower()
        self.pooling = pooling
        self.model, self.preprocess_input, self.target_size = self._build_model()

    def _build_model(self):
        if self.model_name in {"inception", "inceptionv3", "inception_v3"}:
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.applications.inception_v3 import preprocess_input

            model = InceptionV3(weights="imagenet", include_top=False, pooling=self.pooling)
            return model, preprocess_input, (299, 299)

        if self.model_name in {"vgg", "vgg16"}:
            from tensorflow.keras.applications import VGG16
            from tensorflow.keras.applications.vgg16 import preprocess_input

            model = VGG16(weights="imagenet", include_top=False, pooling=self.pooling)
            return model, preprocess_input, (224, 224)

        raise ValueError("model_name must be 'inception_v3' or 'vgg16'")

    def extract(self, image_paths, batch_size=32):
        features = []
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            batch = load_batch(batch_paths, target_size=self.target_size) * 255.0
            batch = self.preprocess_input(batch)
            features.append(self.model.predict(batch, verbose=0))
        return np.vstack(features)

    def save(self, features, save_path):
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.save(save_path, features)

    def load(self, path):
        return np.load(path, allow_pickle=True)
