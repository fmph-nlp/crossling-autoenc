from docopt import docopt
import numpy as np
import os
from os import path
import sys
import math

from keras.preprocessing import text

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
from torch.nn import Parameter


class Autoencoder(nn.Module):
    def __init__(self, n_vocab1, n_vocab2, n_hidden):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(n_vocab1, n_hidden).double(),
            nn.Sigmoid())
        self.encoder2 = nn.Sequential(
            nn.Linear(n_vocab2, n_hidden).double(),
            nn.Sigmoid())
        self.decoder1 = nn.Sequential(
            nn.Linear(n_hidden, n_vocab1).double(),
            nn.Sigmoid())
        self.decoder2 = nn.Sequential(
            nn.Linear(n_hidden, n_vocab2).double(),
            nn.Sigmoid())

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x1 = self.decoder1(x1)
        x2 = self.encoder2(x2)
        x2 = self.decoder2(x2)
        return x1, x2


def main():
    args = docopt("""
    Usage:
        autoenc.py [options] <corpus1> <corpus2> <outdir>

    Example:
        autoenc.py eubookshop.tsv my_model
    """)

    corpus1_file = args['<corpus1>']
    corpus2_file = args['<corpus2>']
    out_dir = args['<outdir>']

    print('Loading language 1 data...')
    x1_train, x1_test = read_corpus(corpus1_file, .1, 4000)
    print('Loading language 2 data...')
    x2_train, x2_test = read_corpus(corpus2_file, .1, 4000)

    print(x1_train.shape)
    print(x1_test.shape)

    vocab1_size = x1_train.shape[1]
    vocab2_size = x2_train.shape[1]

    n_train = x1_train.shape[0]

    # this is the size of our encoded representations
    encoding_dim = 40
    batch_size = 5
    num_epochs = 20
    learning_rate = 0.0003

    model = Autoencoder(vocab1_size, vocab2_size, encoding_dim)
    criterion = nn.BCELoss()

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              momentum=0.3)

    def train_step(begin, end):
        x1 = np.any(x1_train[begin:end, :], axis=0).astype(float)
        x2 = np.any(x2_train[begin:end, :], axis=0).astype(float)
        x1 = Variable(torch.from_numpy(x1))
        x2 = Variable(torch.from_numpy(x2))

        # Forward pass
        out1, out2 = model(x1, x2)
        loss1 = criterion(out1, x1)
        loss2 = criterion(out2, x1)
        loss3 = criterion(out1, x2)
        loss4 = criterion(out2, x2)
        loss = loss1 + loss2 + loss3 + loss4

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if i % 500 == 0:
            print_stat_perm(epoch, num_epochs, i, n_train, loss.data[0])
        elif i % 10 == 0:
            print_stat(epoch, num_epochs, i, n_train, loss.data[0])


    for epoch in range(num_epochs):
        for i in range(0, n_train, batch_size):
            train_step(i, i + batch_size)
        leftover_beg = n_train - n_train % batch_size
        if leftover_beg != n_train:
            train_step(leftover_beg, n_train)


    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), path.join(out_dir, 'autoenc.pth'))


def print_stat(epoch, num_epochs, i, n_train, loss, line_end='\r'):
    print('epoch [{}/{}], example [{}/{}], loss:{:.4f}'
        .format(epoch + 1, num_epochs, i, n_train, loss), end=line_end)
    sys.stdout.flush()


def print_stat_perm(epoch, num_epochs, i, n_train, loss):
    print_stat(epoch, num_epochs, i, n_train, loss, line_end='\n')


def read_corpus(corpus_file, train_factor, num_words):
    lines = []
    with open(corpus_file) as corpus:
        for line in corpus:
            lines.append(line.strip())

    train_size = int(len(lines) * train_factor)
    train_lines = lines[:train_size]
    test_lines = lines[train_size:int(train_size * 1.2)]

    tokenizer = text.Tokenizer(num_words=num_words,
                              lower=False,
                              oov_token='<unk>')

    tokenizer.fit_on_texts(train_lines)
    x_train = tokenizer.texts_to_matrix(train_lines, mode='binary')
    x_test = tokenizer.texts_to_matrix(test_lines, mode='binary')
    return x_train, x_test


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


if __name__ == '__main__':
    main()
