import logging
import sys

import pickle
import numpy as np
import lasagne
import theano
import theano.tensor as T

IMG_COLORS = 3
IMG_SIZE = 32
TRAIN_EPOCHS = 500
MINIBATCH_SIZE = 500


def load_single_dataset(filename):
    with open(filename, "rb") as f:
        file_data = pickle.load(f, encoding='bytes')

    part_data = np.append(np.empty(shape=0, dtype=np.uint8), file_data[b'data'])
    part_labels = np.append(np.empty(shape=0, dtype=np.uint8), file_data[b'labels'])
    return part_data.reshape((-1, IMG_COLORS, IMG_SIZE, IMG_SIZE)).astype(np.int8), part_labels.astype(np.int8)


def load_dataset():
    all_data = np.empty(shape=(0, IMG_COLORS, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    all_labels = np.empty(shape=0, dtype=np.uint8)

    for i in range(1, 6):
        chunk_data, chunk_labels = load_single_dataset("data/data_batch_{}".format(i))
        all_data = np.append(all_data, chunk_data)
        all_labels = np.append(all_labels, chunk_labels)

    return all_data.reshape((-1, IMG_COLORS, IMG_SIZE, IMG_SIZE)), all_labels.astype(np.int8)


def build_network(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, IMG_COLORS, IMG_SIZE, IMG_SIZE), input_var=input_var)

    # Convolutional layer with IMG_SIZE kernels of size 5x5.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=IMG_SIZE, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with IMG_SIZE 5x5 kernels
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=IMG_SIZE, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    # And another 2x2 pooling:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def save_network(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lasagne.layers.get_all_param_values(net), f)


def load_network(filename, input_v=None):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    net = build_network(input_v)
    lasagne.layers.set_all_param_values(net, data)
    return net


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        # yield inputs[1:5], targets[1:5] # TODO: do not use this! never!


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s @%(asctime)s: %(message)s', level=logging.DEBUG)

    logging.info("Loading the training patterns")
    train_data, train_labels = load_dataset()
    if len(train_data) != len(train_labels):
        logging.error("Train data and train labels don't match (%d != %d)", len(train_data), len(train_labels))
        sys.exit(1)
    logging.info("Loaded %d training patterns", len(train_labels))

    logging.info("Loading the test patterns")
    test_data, test_labels = load_single_dataset("data/test_batch")
    if len(test_data) != len(test_labels):
        logging.error("Train data and train labels don't match (%d != %d)", len(test_data), len(test_labels))
        sys.exit(1)
    logging.info("Loaded %d testing patterns", len(test_labels))

    logging.info("Creating Theano input and target variables")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    logging.info("Building the network")
    network = build_network(input_var)

    logging.info("Creating the loss expression")
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    logging.info("Creating the update expression")
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    logging.info("Creating the test-loss expression")
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    logging.info("Creating the test accuracy expression")
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    logging.info("Compiling the train function (Theano)")
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    logging.info("Compiling the validation function (Theano)")
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    logging.info("Starting the training loop")
    for epoch in range(TRAIN_EPOCHS):
        logging.info("Epoch #%d", epoch)

        logging.info("Passing over the training data")
        train_err = 0
        train_batches = 0
        for batch_id, (inputs, targets) in enumerate(iterate_minibatches(train_data, train_labels, MINIBATCH_SIZE, shuffle=True)):
            logging.info("Batch %d in epoch #%d", batch_id, epoch)
            train_err += train_fn(inputs, targets)
            train_batches += 1

        logging.info("Passing over the validation data")
        val_err = 0
        val_acc = 0
        val_batches = 0
        for inputs, targets in iterate_minibatches(test_data, test_labels, MINIBATCH_SIZE, shuffle=False):
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        logging.info("Training loss:\t%f", train_err / train_batches)
        logging.info("Validation loss:\t%f", val_err / val_batches)
        logging.info("Validation accuracy:\t%f%%", val_acc / val_batches * 100)

    logging.info("Training finished")

    network_filename = "network.dat"
    logging.info("Saving the model into '%s'", network_filename)
    save_network(network, network_filename)

    logging.info("Reading it just to be sure")
    input_var2 = T.tensor4('inputs2')
    network2 = load_network(network_filename, input_var2)
    logging.info("Finished! :)")
