import os
from os import path
import sys
import math

from docopt import docopt

import numpy as np
from numpy.random import shuffle

from keras.preprocessing import text

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
from torch.nn import Parameter

import statistics as stats
from autoencoder import BilingualAutoencoder


DEBUG = True


def main():
    args = docopt("""
    Usage:
        autoenc.py [options] <corpus1> <corpus2> <outdir>

    Example:
        autoenc.py sk.lemmas sl.lemmas my_saved_model

    Options:
        --corr       Whether to use the correlation term.
    """)

    corpus1_file = args['<corpus1>']
    corpus2_file = args['<corpus2>']
    out_dir = args['<outdir>']

    corr = args['--corr']

    # Load data
    if DEBUG:
        print('Loading language 1 data...')
        x1_train, x1_test = read_corpus(corpus1_file, .1, 4000)
        print('Loading language 2 data...')
        x2_train, x2_test = read_corpus(corpus2_file, .1, 4000)
    else:
        print('Loading language 1 data...')
        x1_train, x1_test = read_corpus(corpus1_file, .8, 12000)
        print('Loading language 2 data...')
        x2_train, x2_test = read_corpus(corpus2_file, .8, 12000)

    print(x1_train.shape)
    print(x1_test.shape)

    vocab1_size = x1_train.shape[1]
    vocab2_size = x2_train.shape[1]

    n_train = x1_train.shape[0]
    n_test = x1_test.shape[0]

    # Set hyperparameters
    encoding_dim = 40
    batch_size = 5
    num_epochs = 3
    learning_rate = 0.00001
    momentum = 0.0
    corr_lambda = 0.085

    # Train the model
    model = BilingualAutoencoder(vocab1_size, vocab2_size, encoding_dim)
    criterion = nn.BCELoss()

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum)

    def train_step(begin, end):
        x1 = np.any(x1_train[begin:end, :], axis=0).astype(float)
        x2 = np.any(x2_train[begin:end, :], axis=0).astype(float)
        x1 = Variable(torch.from_numpy(x1))
        x2 = Variable(torch.from_numpy(x2))

        # Forward pass
        out1, out2 = model(x1, x2)
        if corr:
            loss = loss_corr(x1, x2, out1, out2,
                                      criterion, model, corr_lambda)
        else:
            loss = loss_combined(x1, x2, out1, out2, criterion)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if i % 500 == 0:
            losses = []
            # Pick 10 test samples for dev loss
            for j in np.random.choice(n_test, 10):
                xt1 = Variable(torch.from_numpy(x1_test[j, :]))
                xt2 = Variable(torch.from_numpy(x2_test[j, :]))
                outt1, outt2 = model(xt1, xt2)
                if corr:
                    losses.append(loss_corr(x1, x2, outt1, outt2,
                                            criterion, model,
                                            corr_lambda).data[0])
                else:
                    losses.append(loss_combined(xt1, xt2,
                                                outt1, outt2,
                                                criterion).data[0])
            dev_loss_avg = sum(losses) / float(len(losses))

            print_stat_perm(epoch, num_epochs, i, n_train,
                            loss.data[0], dev_loss_avg)
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


def loss_combined(x1, x2, y1, y2, criterion):
    loss1 = criterion(y1, x1)
    loss2 = criterion(y2, x1)
    loss3 = criterion(y1, x2)
    loss4 = criterion(y2, x2)
    return loss1 + loss2 + loss3 + loss4


def loss_corr(x1, x2, y1, y2, criterion, model, corr_lambda):
    loss1 = criterion(y1, x2)
    loss2 = criterion(y2, x1)
    corr_term = corr_lambda * stats.pearsonr(model.encoder1(x1),
                                             model.encoder2(x2))
    return loss1 + loss2 - corr_term



def print_stat(epoch, num_epochs, i, n_train, loss, line_end='\r'):
    print('epoch [{}/{}], example [{}/{}], loss:{:.4f}'
        .format(epoch + 1, num_epochs, i, n_train, loss),
                end=line_end)
    sys.stdout.flush()


def print_stat_perm(epoch, num_epochs, i, n_train, loss, devloss):
    print_stat(epoch, num_epochs, i, n_train, loss, line_end='\n')
    print('dev_loss:{:.4f}'.format(devloss))


def read_corpus(corpus_file, train_factor, num_words):
    lines = []
    with open(corpus_file) as corpus:
        for line in corpus:
            lines.append(line.strip())

    train_size = int(len(lines) * train_factor)
    train_lines = lines[:train_size]
    if DEBUG:
        test_lines = lines[train_size:int(train_size * 1.2)]
    else:
        test_lines = lines[train_size:]

    tokenizer = text.Tokenizer(num_words=num_words,
                              lower=False,
                              oov_token='<unk>')

    tokenizer.fit_on_texts(train_lines)
    x_train = tokenizer.texts_to_matrix(train_lines, mode='binary')
    x_test = tokenizer.texts_to_matrix(test_lines, mode='binary')

    return x_train, x_test


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


if __name__ == '__main__':
    main()
