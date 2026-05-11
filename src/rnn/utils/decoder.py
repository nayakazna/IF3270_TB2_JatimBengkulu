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
        logits_or_probs = self.output_projection.forward(recurrent_output)
        return logits_or_probs

    def generate(self, feature):
        feature = np.asarray(feature)
        generated = []
        prefix = [self.vocabulary.start_id]

        for _ in range(self.max_length):
            token_array = np.asarray(prefix, dtype=np.int64)
            probs = self.forward_teacher_forcing(feature, token_array)
            next_distribution = probs[-1]
            if not np.allclose(np.sum(next_distribution), 1.0, atol=1e-5):
                next_distribution = softmax(next_distribution)

            next_id = int(np.argmax(next_distribution))
            if next_id == self.vocabulary.end_id:
                break
            if next_id != self.vocabulary.pad_id:
                generated.append(next_id)
            prefix.append(next_id)

        return self.vocabulary.decode(generated, skip_special_tokens=True)
