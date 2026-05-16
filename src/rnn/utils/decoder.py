import numpy as np

from src.rnn.layers.activation import softmax
from src.rnn.utils.sequence import prepend_feature_timestep


class GreedyCaptionDecoder:
    def __init__(
        self,
        feature_projection,
        embedding,
        recurrent_layer,
        output_projection,
        vocabulary,
        max_length=30,
    ):
        self.feature_projection = feature_projection
        self.embedding = embedding
        self.recurrent_layer = recurrent_layer
        self.output_projection = output_projection
        self.vocabulary = vocabulary
        self.max_length = max_length

    def forward_teacher_forcing(self, features, token_inputs):
        projected = self.feature_projection.forward(features)
        embeddings = self.embedding.forward(token_inputs)
        decoder_input = prepend_feature_timestep(projected, embeddings)
        recurrent_output = self.recurrent_layer.forward(decoder_input)

        if recurrent_output.ndim == 2 and recurrent_output.shape[0] == decoder_input.shape[0]:
            recurrent_output = recurrent_output[:-1, :]
        elif recurrent_output.ndim == 3 and recurrent_output.shape[1] == decoder_input.shape[1]:
            recurrent_output = recurrent_output[:, :-1, :]

        logits_or_probs = self.output_projection.forward(recurrent_output)
        return logits_or_probs

    def generate(self, feature):
        return self.generate_batch(np.asarray(feature)[np.newaxis, :])[0]

    def generate_batch(self, features):
        features = np.asarray(features)
        if features.ndim == 1:
            features = features[np.newaxis, :]
        elif features.ndim != 2:
            raise ValueError("features must have shape (D,) or (N, D)")

        batch_size = features.shape[0]
        prefixes = np.full((batch_size, 1), self.vocabulary.start_id, dtype=np.int64)
        generated = [[] for _ in range(batch_size)]
        finished = np.zeros(batch_size, dtype=bool)

        for _ in range(self.max_length):
            probs = self.forward_teacher_forcing(features, prefixes)
            next_distribution = probs[:, -1, :] if probs.ndim == 3 else probs[-1][np.newaxis, :]
            distribution_sums = np.sum(next_distribution, axis=-1, keepdims=True)
            if not np.allclose(distribution_sums, 1.0, atol=1e-5):
                next_distribution = softmax(next_distribution)

            next_ids = np.argmax(next_distribution, axis=-1).astype(np.int64)
            next_ids = np.where(finished, self.vocabulary.pad_id, next_ids)

            for index, next_id in enumerate(next_ids):
                if finished[index]:
                    continue
                if next_id == self.vocabulary.end_id:
                    finished[index] = True
                elif next_id != self.vocabulary.pad_id:
                    generated[index].append(int(next_id))

            prefixes = np.concatenate([prefixes, next_ids[:, np.newaxis]], axis=1)
            if np.all(finished):
                break

        return [
            self.vocabulary.decode(token_ids, skip_special_tokens=True)
            for token_ids in generated
        ]
