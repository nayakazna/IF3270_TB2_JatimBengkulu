import numpy as np
import nltk

for _pkg in ("punkt", "punkt_tab", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as _meteor_sentence

def compute_bleu4(references_list, hypotheses):
    # Corpus-level BLEU-4.

    # Args:
    # 1. references_list: list of list of reference strings (multiple refs per image)
    # 2. hypotheses: list of hypothesis strings

    refs = [[ref.split() for ref in refs] for refs in references_list]
    hyps = [h.split() for h in hypotheses]
    smoothie = SmoothingFunction().method1
    try:
        return corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    except ZeroDivisionError:
        return 0.0


def compute_meteor(references_list, hypotheses):
    # Average sentence-level METEOR
    scores = []
    for refs, hyp in zip(references_list, hypotheses):
        refs_tok = [r.split() for r in refs]
        hyp_tok = hyp.split()
        score = _meteor_sentence(refs_tok, hyp_tok)
        scores.append(score)
    return float(np.mean(scores))
