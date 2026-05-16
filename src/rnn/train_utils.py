import argparse
import json
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np

from src.rnn.utils.caption_preprocessing import (
    build_vocabulary,
    load_captions_file,
    prepare_decoder_inputs,
    save_vocabulary,
)
from src.rnn.utils.feature_extractor import FeatureExtractor


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_image_id(value):
    return os.path.basename(value.split("#", 1)[0].strip())


def load_split_ids(path):
    if path is None:
        return None

    ids = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            ids.append(normalize_image_id(line.split(",", 1)[0].split()[0]))
    return set(ids)


def flatten_caption_items(captions_by_image, allowed_ids=None):
    image_ids = []
    captions = []

    for image_id, image_captions in captions_by_image.items():
        image_id = normalize_image_id(image_id)
        if allowed_ids is not None and image_id not in allowed_ids:
            continue
        for caption in image_captions:
            image_ids.append(image_id)
            captions.append(caption)

    return image_ids, captions


def split_without_files(captions_by_image, validation_ratio, seed):
    image_ids = np.array(sorted(normalize_image_id(image_id) for image_id in captions_by_image.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * validation_ratio))
    test_count = val_count
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count + test_count:])
    return train_ids, val_ids


def resolve_image_path(image_dir, image_id):
    candidates = [
        Path(image_dir) / image_id,
        Path(image_dir) / "Images" / image_id,
        Path(image_dir) / "Flicker8k_Dataset" / image_id,
        Path(image_dir) / "Flickr8k_Dataset" / image_id,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Image file not found for {image_id} under {image_dir}")


def load_feature_cache(path):
    if not path or not os.path.exists(path):
        return {}

    data = np.load(path, allow_pickle=True)
    ids = data["image_ids"]
    features = data["features"]
    return {str(image_id): features[index] for index, image_id in enumerate(ids)}


def save_feature_cache(path, feature_map):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_ids = np.array(sorted(feature_map.keys()))
    features = np.stack([feature_map[image_id] for image_id in image_ids], axis=0)
    np.savez_compressed(path, image_ids=image_ids, features=features)


def ensure_features(image_ids, image_dir, cache_path, cnn_model_name, batch_size):
    feature_map = load_feature_cache(cache_path)
    missing_ids = sorted(set(image_ids) - set(feature_map.keys()))

    if missing_ids:
        extractor = FeatureExtractor(model_name=cnn_model_name)
        image_paths = [resolve_image_path(image_dir, image_id) for image_id in missing_ids]
        extracted = extractor.extract(image_paths, batch_size=batch_size)
        for image_id, feature in zip(missing_ids, extracted):
            feature_map[image_id] = feature
        save_feature_cache(cache_path, feature_map)

    return feature_map


def feature_matrix(image_ids, feature_map):
    return np.stack([feature_map[image_id] for image_id in image_ids], axis=0).astype(np.float32)


def build_decoder_model(decoder_type, feature_dim, vocab_size, max_len, embed_dim, hidden_size, recurrent_layers, dropout):
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, LSTM, Lambda, SimpleRNN

    feature_input = Input(shape=(feature_dim,), name="features")
    token_input = Input(shape=(max_len,), dtype="int32", name="tokens")

    projected_feature = Dense(embed_dim, name="feature_projection")(feature_input)
    feature_timestep = Lambda(lambda x: tf.expand_dims(x, axis=1), name="feature_timestep")(projected_feature)
    token_embeddings = Embedding(vocab_size, embed_dim, name="embedding")(token_input)
    x = Concatenate(axis=1, name="pre_inject_sequence")([feature_timestep, token_embeddings])

    recurrent_cls = LSTM if decoder_type == "lstm" else SimpleRNN
    for index in range(recurrent_layers):
        x = recurrent_cls(
            hidden_size,
            return_sequences=True,
            dropout=dropout,
            name=f"{decoder_type}_{index + 1}",
        )(x)

    x = Lambda(lambda value: value[:, :-1, :], name="shifted_outputs")(x)
    output = Dense(vocab_size, activation="softmax", name="output_projection")(x)
    model = Model(inputs=[feature_input, token_input], outputs=output, name=f"{decoder_type}_decoder")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def save_history(history, path):
    payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_config(config, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def train_decoder(decoder_type, args):
    captions_by_image = load_captions_file(args.captions_path, delimiter=args.delimiter, skip_header=args.skip_header)

    train_ids = load_split_ids(args.train_split)
    val_ids = load_split_ids(args.val_split)
    if train_ids is None or val_ids is None:
        train_ids, val_ids = split_without_files(captions_by_image, args.validation_ratio, args.seed)

    train_image_ids, train_captions = flatten_caption_items(captions_by_image, train_ids)
    val_image_ids, val_captions = flatten_caption_items(captions_by_image, val_ids)

    if not train_captions:
        raise ValueError("No training captions found")
    if not val_captions:
        raise ValueError("No validation captions found")

    vocabulary = build_vocabulary(train_captions, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    train_tokens, train_targets = prepare_decoder_inputs(train_captions, vocabulary, max_len=args.max_len)
    val_tokens, val_targets = prepare_decoder_inputs(val_captions, vocabulary, max_len=train_tokens.shape[1])

    all_image_ids = set(train_image_ids) | set(val_image_ids)
    feature_map = ensure_features(
        all_image_ids,
        args.image_dir,
        args.feature_cache,
        args.cnn_model,
        args.feature_batch_size,
    )

    train_features = feature_matrix(train_image_ids, feature_map)
    val_features = feature_matrix(val_image_ids, feature_map)

    output_dir = Path(args.output_dir) / decoder_type
    output_dir.mkdir(parents=True, exist_ok=True)
    save_vocabulary(vocabulary, output_dir / "vocabulary.json")

    feature_dim = train_features.shape[1]
    vocab_size = len(vocabulary.word_to_id)
    hidden_sizes = parse_int_list(args.hidden_sizes)
    recurrent_layer_counts = parse_int_list(args.recurrent_layers)

    for recurrent_layers in recurrent_layer_counts:
        for hidden_size in hidden_sizes:
            run_name = f"{decoder_type}_layers{recurrent_layers}_hidden{hidden_size}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            model = build_decoder_model(
                decoder_type,
                feature_dim,
                vocab_size,
                train_tokens.shape[1],
                args.embed_dim,
                hidden_size,
                recurrent_layers,
                args.dropout,
            )

            callbacks = []
            if args.early_stopping:
                from tensorflow.keras.callbacks import EarlyStopping

                callbacks.append(
                    EarlyStopping(
                        monitor="val_loss",
                        patience=args.patience,
                        restore_best_weights=True,
                    )
                )

            history = model.fit(
                [train_features, train_tokens],
                train_targets[..., np.newaxis],
                validation_data=([val_features, val_tokens], val_targets[..., np.newaxis]),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks,
            )

            model.save_weights(run_dir / "weights.h5")
            save_history(history, run_dir / "history.json")
            save_config(
                {
                    "decoder_type": decoder_type,
                    "feature_dim": int(feature_dim),
                    "vocab_size": int(vocab_size),
                    "max_len": int(train_tokens.shape[1]),
                    "embed_dim": int(args.embed_dim),
                    "hidden_size": int(hidden_size),
                    "recurrent_layers": int(recurrent_layers),
                    "train_caption_count": int(len(train_captions)),
                    "val_caption_count": int(len(val_captions)),
                    "cnn_model": args.cnn_model,
                },
                run_dir / "config.json",
            )


def build_arg_parser(decoder_type):
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--train-split")
    parser.add_argument("--val-split")
    parser.add_argument("--delimiter")
    parser.add_argument("--skip-header", action="store_true")
    parser.add_argument("--feature-cache", default=f"data/features/flickr8k_{decoder_type}_features.npz")
    parser.add_argument("--output-dir", default="data/models/rnn")
    parser.add_argument("--cnn-model", default="inception_v3")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-sizes", default="128,512")
    parser.add_argument("--recurrent-layers", default="1,2,3")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    return parser
