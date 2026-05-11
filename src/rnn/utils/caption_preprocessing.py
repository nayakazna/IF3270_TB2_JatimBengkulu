import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocabulary:
    word_to_id: dict
    id_to_word: dict
    pad_token: str = PAD_TOKEN
    start_token: str = START_TOKEN
    end_token: str = END_TOKEN
    unk_token: str = UNK_TOKEN

    @property
    def pad_id(self):
        return self.word_to_id[self.pad_token]

    @property
    def start_id(self):
        return self.word_to_id[self.start_token]

    @property
    def end_id(self):
        return self.word_to_id[self.end_token]

    @property
    def unk_id(self):
        return self.word_to_id[self.unk_token]

    def encode(self, caption, add_special_tokens=True):
        tokens = clean_caption(caption).split()
        ids = [self.word_to_id.get(token, self.unk_id) for token in tokens]
        if add_special_tokens:
            ids = [self.start_id] + ids + [self.end_id]
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        words = []
        special = {self.pad_token, self.start_token, self.end_token}
        for token_id in token_ids:
            word = self.id_to_word.get(str(int(token_id)), self.unk_token)
            if skip_special_tokens and word in special:
                continue
            words.append(word)
        return " ".join(words)


def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", " ", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def load_captions_file(path, delimiter=None, skip_header=False):
    captions = {}
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if skip_header and line_number == 0:
                continue
            line = line.strip()
            if not line:
                continue

            if delimiter is not None:
                parts = line.split(delimiter, 1)
            elif "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(",", 1)

            if len(parts) != 2:
                continue

            image_id = parts[0].split("#", 1)[0].strip()
            caption = parts[1].strip()
            captions.setdefault(image_id, []).append(caption)

    return captions


def build_vocabulary(captions, min_freq=1, max_vocab_size=None):
    counts = {}
    for caption in captions:
        for token in clean_caption(caption).split():
            counts[token] = counts.get(token, 0) + 1

    words = [word for word, count in counts.items() if count >= min_freq]
    words.sort(key=lambda word: (-counts[word], word))
    if max_vocab_size is not None:
        reserved = 4
        words = words[: max(0, max_vocab_size - reserved)]

    ordered = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN] + words
    word_to_id = {word: idx for idx, word in enumerate(ordered)}
    id_to_word = {str(idx): word for word, idx in word_to_id.items()}
    return Vocabulary(word_to_id=word_to_id, id_to_word=id_to_word)


def save_vocabulary(vocabulary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "word_to_id": vocabulary.word_to_id,
        "id_to_word": vocabulary.id_to_word,
        "pad_token": vocabulary.pad_token,
        "start_token": vocabulary.start_token,
        "end_token": vocabulary.end_token,
        "unk_token": vocabulary.unk_token,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


def load_vocabulary(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return Vocabulary(
        word_to_id={word: int(idx) for word, idx in payload["word_to_id"].items()},
        id_to_word={str(idx): word for idx, word in payload["id_to_word"].items()},
        pad_token=payload.get("pad_token", PAD_TOKEN),
        start_token=payload.get("start_token", START_TOKEN),
        end_token=payload.get("end_token", END_TOKEN),
        unk_token=payload.get("unk_token", UNK_TOKEN),
    )


def pad_sequences(sequences, max_len=None, padding="post", truncating="post", value=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    out = np.full((len(sequences), max_len), value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = list(seq)
        if truncating == "post":
            trunc = seq[:max_len]
        elif truncating == "pre":
            trunc = seq[-max_len:]
        else:
            raise ValueError("truncating must be 'pre' or 'post'")

        if padding == "post":
            out[i, : len(trunc)] = trunc
        elif padding == "pre":
            out[i, -len(trunc) :] = trunc
        else:
            raise ValueError("padding must be 'pre' or 'post'")

    return out


def prepare_decoder_inputs(captions, vocabulary, max_len=None):
    encoded = [vocabulary.encode(caption, add_special_tokens=True) for caption in captions]
    if max_len is None:
        max_len = max(len(seq) for seq in encoded) - 1

    input_sequences = [seq[:-1] for seq in encoded]
    target_sequences = [seq[1:] for seq in encoded]

    inputs = pad_sequences(
        input_sequences,
        max_len=max_len,
        padding="post",
        truncating="post",
        value=vocabulary.pad_id,
    )
    targets = pad_sequences(
        target_sequences,
        max_len=max_len,
        padding="post",
        truncating="post",
        value=vocabulary.pad_id,
    )
    return inputs, targets
