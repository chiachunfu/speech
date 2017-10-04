import os, sys
from glob import iglob, glob

from python_speech_features import mfcc
import numpy as np
import random


def shuffle_every_epoch(data_path):
    """Return a list of shuffled sample file names.
        Args:
            data_path: the folder path of the samples (string)
        Returns:
            alist: list of file names of the samples (string)
    """
    alist = os.listdir(data_path)
    random.shuffle(alist)
    return alist

def next_batch_training(batch_size, trainlist, batch, DIR):
    """Return batch for training.
        Args:
            batch_size: number of samples per batch (int)
            trainlist: a list of samples file names in the folder (string)
            batch: the order of a batch in the epoch (int)
            DIR: the folder path for the samples (string)
        Returns:
            train_inputs: a batch of mfcc training data [batch_size, maxlen] (float)
            train_targets: a batch of target labels in index form and sparse tuple format (int)
            train_seq_len: a batch of training data sequence lengths [batch_size] (int)
            original: a batch of training target data in their original forms. (string)
    """
    inputs_batch = []
    targets = []
    original = []
    seq_len = []
    maxlen = 0
    num_features = 39
    for file in trainlist[batch_size*batch:batch_size*(batch+1)]:
        datafile = np.load(os.path.join(DIR,file))
        inputs_batch.append(datafile['data_in'])
        targets.append(datafile['target'])
        seq_len.append(datafile['seq_len'][0])
        original.append(datafile['original'][0])
        if datafile['seq_len'][0] > maxlen:
            maxlen = datafile['seq_len'][0]
    # Pad the inputs to the maxlen with 0s
    for i in range(batch_size):
        inputs_batch[i] = np.pad(inputs_batch[i],((0,maxlen-len(inputs_batch[i])),(0,0)), mode='constant', constant_values=0)

    train_inputs = np.asarray(inputs_batch)
    train_seq_len = np.asarray(seq_len)
    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from(targets)
    return train_inputs, train_targets, train_seq_len, original

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
