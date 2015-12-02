import logging
import sys
import argparse
import os

import pickle
import numpy as np

DATA_DIR = "data"
DATA_FILENAME = "cifar-10-python.tar.gz"
DATA_PATH = DATA_DIR + "/" + DATA_FILENAME

BATCH_NUMBER = 6
BATCH_FILENAME = "data_batch_{}"
BATCH_PATH = DATA_DIR + "/" + BATCH_FILENAME
BATCH_URL = "http://www.cs.toronto.edu/~kriz/" + DATA_FILENAME


def check_or_download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    batch_exists = True
    for i in range(1, BATCH_NUMBER):
        if not os.path.exists(BATCH_PATH.format(i)):
            batch_exists = False
            break

    if not batch_exists:
        import urllib.request
        import tarfile

        logging.info("Downloading the requested dataset: " + BATCH_URL)

        requested_files = [BATCH_FILENAME.format(i) for i in range(1, BATCH_NUMBER)] + ['test_batch']

        # copypasta from stackoverflow
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                import math

                ts_l = math.floor(math.log(totalsize, 2) / 10.0)
                sf_l = math.floor(math.log(readsofar + 1, 2) / 10.0)

                def get_s_from_l(l):
                    if l == 0:
                        return 'B'
                    elif l == 1:
                        return 'kB'
                    elif l == 2:
                        return 'MB'
                    elif l == 3:
                        return 'GB'

                ts_s = get_s_from_l(ts_l)
                sf_s = get_s_from_l(sf_l)

                ts_l *= 10
                sf_l *= 10

                percent = readsofar * 1e2 / totalsize

                s = "\r%5.1f%% %*d %s / %d %s" % (percent, len(str(totalsize)), readsofar / (2 ** sf_l), sf_s, totalsize / (2 ** ts_l), ts_s)
                sys.stderr.write(s)
                if readsofar >= totalsize:  # near the end
                    sys.stderr.write("\n")
            else:  # total size is unknown
                sys.stderr.write("read %d\n" % (readsofar,))

        urllib.request.urlretrieve(BATCH_URL, DATA_PATH, reporthook)
        with tarfile.open(DATA_PATH) as tf:
            for member in tf.getmembers():
                if member.name.split("/")[-1] in requested_files:
                    tf.extract(member, DATA_DIR)

                    # rename the fucker
                    os.rename(DATA_DIR + "/" + member.name, DATA_DIR + "/" + member.name.split("/")[-1])


def load_single_dataset(filename):
    check_or_download_dataset()

    with open(filename, "rb") as f:
        file_data = pickle.load(f, encoding='bytes')

    part_data = np.append(np.empty(shape=0, dtype=np.uint8), file_data[b'data'])
    part_labels = np.append(np.empty(shape=0, dtype=np.uint8), file_data[b'labels'])
    return part_data.reshape((-1, config.img_colors, config.img_size, config.img_size)).astype(np.int8), part_labels.astype(np.int8)


def load_dataset():
    all_data = np.empty(shape=(0, config.img_colors, config.img_size, config.img_size), dtype=np.uint8)
    all_labels = np.empty(shape=0, dtype=np.uint8)

    for i in range(1, BATCH_NUMBER):
        chunk_data, chunk_labels = load_single_dataset(BATCH_PATH.format(i))
        all_data = np.append(all_data, chunk_data)
        all_labels = np.append(all_labels, chunk_labels)

    return all_data.reshape((-1, config.img_colors, config.img_size, config.img_size)), all_labels.astype(np.int8)


def build_network(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, config.img_colors, config.img_size, config.img_size), input_var=input_var)

    # Convolutional layer with IMG_SIZE kernels of size 5x5.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=config.img_size, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with IMG_SIZE 5x5 kernels
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=config.img_size, filter_size=(5, 5),
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
        # yield inputs[1:5], targets[1:5] # TODO: do not use this! ever!


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s @%(asctime)s: %(message)s', level=logging.DEBUG)

    logging.info("Parsing arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", nargs=1, default=["cpu"], help="Train on this device")
    parser.add_argument("-t", "--trainepochs", default=500, type=int, help="Number of train epochs")
    parser.add_argument("-b", "--minibatch", default=100, type=int, help="Size of the minibatch")
    parser.add_argument("-m", "--mode", default="FAST_RUN", help="Theano run mode")
    parser.add_argument("-f", "--floatX", default="float32", help="Theano floatX mode")

    global config
    config = parser.parse_args()
    config.device = config.device[0]
    config.img_colors = 3
    config.img_size = 32

    logging.info("Setting environmental variables for Theano")
    os.environ["THEANO_FLAGS"] = "mode={},device={},floatX={}".format(config.mode, config.device, config.floatX)

    logging.info("Importing Theano and Lasagne")
    import theano
    import theano.tensor as T
    import lasagne

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
    for epoch in range(config.trainepochs):
        logging.info("Epoch #%d", epoch)

        logging.info("Passing over the training data")
        train_err = 0
        train_batches = 0
        for batch_id, (inputs, targets) in enumerate(iterate_minibatches(train_data, train_labels, config.minibatch, shuffle=True)):
            logging.info("Batch %d in epoch #%d", batch_id, epoch)
            train_err += train_fn(inputs, targets)
            train_batches += 1

        logging.info("Passing over the validation data")
        val_err = 0
        val_acc = 0
        val_batches = 0
        for inputs, targets in iterate_minibatches(test_data, test_labels, config.minibatch, shuffle=False):
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
