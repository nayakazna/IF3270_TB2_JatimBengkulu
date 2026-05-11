import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rnn.train_utils import build_arg_parser, train_decoder


if __name__ == "__main__":
    train_decoder("rnn", build_arg_parser("rnn").parse_args())
