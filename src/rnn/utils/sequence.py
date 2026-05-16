import numpy as np


def prepend_feature_timestep(projected_features, token_embeddings):
    projected_features = np.asarray(projected_features)
    token_embeddings = np.asarray(token_embeddings)

    squeeze_batch = False
    if token_embeddings.ndim == 2:
        token_embeddings = token_embeddings[np.newaxis, ...]
        squeeze_batch = True
    elif token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape (T, D) or (N, T, D)")

    if projected_features.ndim == 1:
        projected_features = projected_features[np.newaxis, :]
    elif projected_features.ndim != 2:
        raise ValueError("projected_features must have shape (D,) or (N, D)")

    if token_embeddings.shape[0] == 1 and projected_features.shape[0] > 1:
        token_embeddings = np.repeat(token_embeddings, projected_features.shape[0], axis=0)
        squeeze_batch = False

    if projected_features.shape[0] == 1 and token_embeddings.shape[0] > 1:
        projected_features = np.repeat(projected_features, token_embeddings.shape[0], axis=0)

    if projected_features.shape[0] != token_embeddings.shape[0]:
        raise ValueError("Batch size of projected_features and token_embeddings must match")
    if projected_features.shape[1] != token_embeddings.shape[2]:
        raise ValueError("Feature projection size must match embedding dimension")

    feature_step = projected_features[:, np.newaxis, :]
    out = np.concatenate([feature_step, token_embeddings], axis=1)
    return out[0] if squeeze_batch else out
