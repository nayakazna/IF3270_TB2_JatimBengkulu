from src.rnn.layers.activation import softmax
from src.rnn.layers.dense import Dense
from src.rnn.layers.embedding import Embedding
from src.rnn.layers.lstm import LSTM
from src.rnn.layers.simple_rnn import SimpleRNN


def layer_from_keras(layer):
    class_name = layer.__class__.__name__.lower()

    if class_name == "embedding":
        return Embedding.from_keras(layer)

    if class_name == "simplernn":
        return SimpleRNN.from_keras(layer)

    if class_name == "lstm":
        return LSTM.from_keras(layer)

    if class_name == "dense":
        activation_name = getattr(getattr(layer, "activation", None), "__name__", None)
        activation = softmax if activation_name == "softmax" else None
        return Dense.from_keras(layer, activation=activation)

    raise ValueError(f"Unsupported Keras layer type: {layer.__class__.__name__}")
