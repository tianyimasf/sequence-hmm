from time import time
import numpy as np
import os
import sys


# File: nlputil.py
# Purpose: A collection of utility methods for working with text data.
# Author: Adam Purtee (starter code for CSC 246 Spring 2021 third Project)
#
#
# This file contains three kinds of methods for each of the following tasks:
#  o building a vocabulary -- this obtains a dictionary from tokens to integers
#  o converting to integers -- this converts a sample to a sequence of integers
#  o converting to one-hot -- this converts a sample to a sequence of one-hot vectors
#                             (i.e., this maps strings to matrices).
#
# There are parallel methods included for working with either word-based or character-based models.
# In both cases, unknown words will always map to integer value 0.

# Note -- File IO is slow, so it's best if you can keep as much data in RAM as possible.
# Converting to a word-based one-hot representation is EXTREMELY memory intensive (many huge vectors of ints).
# Converting to a character-based one-hot representation is also expensive (more vectors, but smaller dimension).
# The most memory efficient implementation would probably read all of the data into RAM, leave it as a gigantic
# python list of strings, and do vocabulary lookup on the fly during learning/inference.
# The fastest implementation would precompute as much stuff as possible.


# A vocabulary for this project is a dictionary from tokens to unique integers.
#
# This will simply assign each unique word in the data a unique integer, starting with one
# and increasing until all words are processed.  The exact results depend on the order in
# which the paths are presented to the method.  Renaming the files may also change the order.
#
# paths -- a list of paths to directories containing data.  paths represented as strings)
def build_vocab_words(paths, sample_size):
    vocab = {}
    int_to_word_map = {}
    nextValue = 0
    count = 0
    sizes = [sample_size/2, sample_size-sample_size/2]
    for idx, path in enumerate(paths):
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as fh:
                sequence = fh.read()
                tokenized_seq = sequence.split()
                if len(tokenized_seq) < 100 and len(tokenized_seq) > 0:
                    for token in tokenized_seq:
                        if token not in vocab:
                            vocab[token] = nextValue
                            int_to_word_map[nextValue] = token
                            nextValue += 1
                    count += 1
            if count >= sizes[idx]:
                count = 0
                break
    print("finished")
    return vocab, int_to_word_map


def translate_int_to_words(sample, int_to_word_map):
    answer = []
    for i in range(0, len(sample)):
        answer.append(int_to_word_map.get(sample[i]))
    return answer


def build_test_samples(paths, sample_size, vocab):
    samples = []
    sizes = [sample_size/2, sample_size-sample_size/2]
    for idx, path in enumerate(paths):
        l = os.listdir(path)
        for i in range(0, int(sizes[idx])):
            with open(os.path.join(path, l[i]), encoding='utf-8') as fh:
                sample = convert_words_to_ints(fh.read(), vocab)
                sample = [i for i in sample if i != -1]  # Remove all UNK
                samples.append(sample)
    return samples

# Same as above, but for character models.


def build_vocab_chars(paths, sample_size):
    vocab = {}
    nextValue = 0
    count = 0
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as fh:
                sequence = fh.read()
                if len(sequence) < 500 and len(sequence) > 0:
                    for character in sequence:
                        if character not in vocab:
                            vocab[character] = nextValue
                            nextValue += 1
                    count += 1
            if count == sample_size:
                break
        if count == sample_size:
            break
    print("finished")
    return vocab


# Sample is a plain string - not a list -- UNK token has value zero --> changing it to -1
# Convert the sample to a integer representation, which is an Nx1 array of ints,
# where N is the number of tokens in the sequence.
def convert_words_to_ints(sample, vocab):
    sequence = sample.split()
    answer = np.zeros(len(sequence), dtype=np.int64)
    for n, token in enumerate(sequence):
        answer[n] = vocab.get(token, -1)
    return answer


def convert_test_seq_into_ints(sample, vocab):
    sequence = sample.split()
    answer = []
    for token in sequence:
        if vocab.get(token, -1) is not None:
            answer.append(vocab.get(token, -1))
    return answer

# Same as above, but for characters.


def convert_chars_to_ints(sample, vocab):
    answer = np.zeros(len(sample), dtype=np.uint)
    for n, token in enumerate(sample):
        answer[n] = vocab.get(token, 0)
    return answer


# Sample is a plain string - not a list -- UNK token has value zero.
# Convert the sample to a one-hot representation, which is an NxV matrix,
# where N is the number of tokens in the sequence and V is the vocabulary
# size observed on the training data.
def convert_words_to_onehot(sample, vocab):
    sequence = sample.split()
    onehot = np.zeros((len(sequence), len(vocab)+1), dtype=np.uint)
    for n, token in enumerate(sequence):
        onehot[n, vocab.get(token, 0)] = 1
    return onehot

# Same as above, but for characters.


def convert_chars_to_onehot(sample, vocab):
    onehot = np.zeros((len(sample), len(vocab)+1), dtype=np.uint)
    for n, token in enumerate(sample):
        onehot[n, vocab.get(token, 0)] = 1
    return onehot


# Read every file located at given path, convert to one-hot OR integer representation,
# and collect the results into a python list.
def load_and_convert_data_words_to_onehot(paths, vocab):
    data = []
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as fh:
                data.append(convert_words_to_onehot(fh.read(), vocab))
    return data

# Same as above, but uses a character model


def load_and_convert_data_chars_to_onehot(paths, vocab):
    data = []
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as fh:
                data.append(convert_chars_to_onehot(fh.read(), vocab))
    return data


def load_and_convert_data_words_to_ints(paths, vocab, sample_size):
    data = []
    count = 0
    sizes = [sample_size/2, sample_size-sample_size/2]
    for idx, path in enumerate(paths):
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as fh:
                sample = convert_words_to_ints(fh.read(), vocab)
                if len(sample) > 0 and len(sample) < 100:
                    data.append(sample)
                    count += 1
            if count >= sizes[idx]:
                count = 0
                break
    print("finished")
    return data


def load_and_convert_test_data_to_ints(paths, vocab, sample_size):
    data = []
    count = 0
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as fh:
                sample = convert_test_seq_into_ints(fh.read(), vocab)
                if len(sample) > 0 and len(sample) < 100:
                    data.append(sample)
                    count += 1
            if count >= sample_size/2:
                break
        if count >= sample_size/2:
            break
    print("finished")
    return data

# Same as above, but uses a character model


def load_and_convert_data_chars_to_ints(paths, vocab, sample_size):
    data = []
    count = 0
    for path in paths:
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as fh:
                sequence = fh.read()
                if len(sequence) < 500 and len(sequence) > 0:
                    data.append(convert_chars_to_ints(sequence, vocab))
                    count += 1
            if count == sample_size:
                break
        if count == sample_size:
            break
    return data


if __name__ == '__main__':
    print("NLP Util smoketest.")
    paths = ['../data/imdbFor246/train/pos', '../data/imdbFor246/train/neg']
    print("Begin loading vocab... ", end='')
    sys.stdout.flush()
    begin = time()
    vocab = build_vocab_chars(paths)
    end = time()
    print('done in', end-begin, 'seconds.  Found', len(vocab), 'unique tokens.')
    print('Begin loading all data and converting to ints... ', end='')
    sys.stdout.flush()
    begin = time()
    data = load_and_convert_data_chars_to_ints(paths, vocab)
    end = time()
    print('done in', end-begin, 'seconds.')

    print("Data[0] = ", data[0])
    print('Press enter to quit.')
    input()
    print('Quitting.. may take some time to free memory.')
