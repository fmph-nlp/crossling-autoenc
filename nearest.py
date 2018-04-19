import os
from os import path
import sys
import math
from functools import partial

from docopt import docopt
import numpy as np
from keras.preprocessing import text
import torch
from scipy import spatial

import statistics as stats
from autoencoder import BilingualAutoencoder
from train import read_corpus


def main():
    args = docopt("""
    Usage:
        closest.py [options] <corpus1> <corpus2> <vocab_size> <model> <word> <lang> <num_results>

    Example:
        closest.py sk.lemmas sl.lemmas model.pth
    """)

    corpus1_file = args['<corpus1>']
    corpus2_file = args['<corpus2>']
    vocab_size = int(args['<vocab_size>'])
    model_file = args['<model>']
    word = args['<word>']
    lang = args['<lang>']
    num_results = int(args['<num_results>'])

    print('Loading language 1 data...')
    seqs1, tok1 = read_corpus(corpus1_file, vocab_size)
    print('Loading language 2 data...')
    seqs2, tok2 = read_corpus(corpus2_file, vocab_size)

    encoding_dim = 40
    model = BilingualAutoencoder(vocab_size, vocab_size, encoding_dim)
    model.load_state_dict(torch.load(model_file))

    print('Model loaded')

    # Extract the matrices
    w1 = model.encoder1[0].weight.data.numpy().T
    w2 = model.encoder2[0].weight.data.numpy().T

    if lang == 'sl':
        tok = tok2
        w = w2
    else:
        tok = tok1
        w = w1

    w_ind = tok.word_index.get(word)
    wordvec = w[w_ind]

    print('Computing similarities [sk]...')
    find_closest(wordvec, num_results, w1, tok1)

    print('Computing similarities [sl]...')
    find_closest(wordvec, num_results, w2, tok2)


def find_closest(wordvec, num_results, w, tok):
    difs = np.apply_along_axis(partial(cos_difference, wordvec), 1, w)
    idx = np.argsort(difs)
    index_word = {v: k for k, v in tok.word_index.items()}
    for i in range(num_results):
        print('word %2d with similarity %.4f: %s' % (i + 1, 1 - difs[idx[i]], index_word[idx[i]]))


def cos_difference(x, y):
    return spatial.distance.cosine(x, y)


def read_corpus(corpus_file, num_words):
    lines = []
    with open(corpus_file) as corpus:
        for line in corpus:
            lines.append(line.strip())

    tokenizer = text.Tokenizer(num_words=num_words,
                              lower=False,
                              oov_token='<unk>')

    tokenizer.fit_on_texts(lines)
    return tokenizer.texts_to_sequences(lines), tokenizer


if __name__ == '__main__':
    main()
