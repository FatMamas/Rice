import pickle
import numpy as np

import logging
import sys


def load_single_dataset(filename):
    with open(filename, "rb") as f:
        file_data = pickle.load(f, encoding='bytes')
    return file_data[b'data'], file_data[b'labels']


def load_dataset():
    all_data = np.empty(shape=0, dtype=np.uint8)
    all_labels = np.empty(shape=0, dtype=np.uint8)

    for i in range(1,6):
        chunk_data, chunk_labels = load_single_dataset("data/data_batch_{}".format(i))
        all_data = np.append(all_data, chunk_data)
        all_labels = np.append(all_labels, chunk_labels)

    return all_data.reshape((len(all_labels), 3072)), all_labels


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s @%(asctime)s: %(message)s', level=logging.DEBUG)

    # load training data
    train_data, train_labels = load_dataset()
    if len(train_data) != len(train_labels):
        logging.error("Train data and train labels don't match (%d != %d)", len(train_data), len(train_labels))
        sys.exit(1)
    logging.info("Loaded %d training patterns", len(train_labels))

    # load testing data
    test_data, test_labels = load_single_dataset("data/test_batch")
    if len(test_data) != len(test_labels):
        logging.error("Train data and train labels don't match (%d != %d)", len(test_data), len(test_labels))
        sys.exit(1)
    logging.info("Loaded %d testing patterns", len(test_labels))
