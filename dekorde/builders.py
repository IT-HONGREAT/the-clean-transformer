from typing import List, Tuple
import torch
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# this is for building an input
def build_I(sents: List[str], tokenizer: Tokenizer, max_length: int, device: torch.device) -> torch.LongTensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    # use zero for the pad token
    seqs = pad_sequences(sequences=seqs, maxlen=max_length, padding="post", value=0)
    return torch.LongTensor(seqs).to(device)


def build_X(gibs: List[str], tokenizer: Tokenizer, max_length: int, device: torch.device) -> torch.LongTensor:
    return build_I(gibs, tokenizer, max_length, device)


def build_Y(kors: List[str], tokenizer: Tokenizer, max_length: int, device: torch.device) -> torch.LongTensor:
    Y = build_I(kors, tokenizer, max_length, device)
    torch.val
    Y_r = Y[:, :-1]  # right-shifted


def build_M(max_length: int, device: torch.device) -> torch.Tensor:
    """
    :param max_length: L
    :param device:
    :return: M (L, L)
    """
    M = torch.tril(torch.ones(size=(max_length, max_length)), diagonal=0)
    return M.to(device)