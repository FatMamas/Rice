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
    return part_data.reshape((-1, config.img_colors, config.img_size, config.img_size)).astype(np.float32)/255.0, part_labels.astype(np.int8)


def load_dataset():
    all_data = np.empty(shape=(0, config.img_colors, config.img_size, config.img_size), dtype=np.float32)
    all_labels = np.empty(shape=0, dtype=np.float32)

    for i in range(1, BATCH_NUMBER):
        chunk_data, chunk_labels = load_single_dataset(BATCH_PATH.format(i))
        all_data = np.append(all_data, chunk_data)
        all_labels = np.append(all_labels, chunk_labels)

    return all_data.reshape((-1, config.img_colors, config.img_size, config.img_size)), all_labels.astype(np.int8)


def save_network(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lasagne.layers.get_all_param_values(net), f)


def load_network(filename, net):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
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


def save_img(img, f_name):
    from PIL import Image
    swapped_img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 0)
    Image.fromarray(swapped_img, 'RGB').save(f_name)


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s @%(asctime)s: %(message)s', level=logging.DEBUG)

    logging.info("Parsing arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", nargs=1, default=["cpu"], help="Train on this device")
    parser.add_argument("-t", "--trainepochs", default=500, type=int, help="Number of train epochs")
    parser.add_argument("-b", "--minibatch", default=100, type=int, help="Size of the minibatch")
    parser.add_argument("-m", "--mode", default="FAST_RUN", help="Theano run mode")
    parser.add_argument("-f", "--floatX", default="float32", help="Theano floatX mode")
    parser.add_argument("-l", "--log", default="log", help="Log directory")
    parser.add_argument("-o", "--output", default="trained", help="Trained model output directory")

    global config
    config = parser.parse_args()
    config.device = config.device[0]
    config.img_colors = 3
    config.img_size = 32

    logging.info("Setting environmental variables for Theano")
    os.environ["THEANO_FLAGS"] = "mode={},device={},floatX={},nvcc.fastmath=True".format(config.mode, config.device, config.floatX)

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
    input_var = T.tensor4('inputs', dtype='float32')
    target_var = T.ivector('targets')

    logging.info("Importing the network module")    # we need to import this AFTER Theano and Lasagne
    from model.mnist import build_network as build_network
    # from model.official import build_cifar_network as build_network
    # from model.conv3 import build_network_3cc as build_network

    logging.info("Building the network")
    network, net_name = build_network(config, input_var)
    logging.info("Network '%s' successfully built", net_name)

    logging.info("Creating the loss expression")
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    logging.info("Creating the update expression")
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.5, momentum=0.3)
    #updates = lasagne.updates.adagrad(loss, params) # this kinda works

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

    logging.info("Creating log file")
    os.makedirs(config.log, exist_ok=True)
    with open('{}/{}.csv'.format(config.log, net_name), 'w') as log_f:
        log_f.write("epoch;trainloss;valloss;valacc\n")

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

            logging.info("Training loss:\t%.10f", train_err / train_batches)
            logging.info("Validation loss:\t%.10f", val_err / val_batches)
            logging.info("Validation accuracy:\t%.10f%%", val_acc / val_batches * 100)

            log_f.write("{};{};{};{}\n".format(epoch, train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100))

    logging.info("Training finished")

    os.makedirs(config.output, exist_ok=True)
    network_filename = "{}/network_{}.dat".format(config.output, net_name)
    logging.info("Saving the model into '%s'", network_filename)
    save_network(network, network_filename)

    # logging.info("Reading it just to be sure")
    # input_var2 = T.tensor4('inputs2')
    # network2 = load_network(network_filename, input_var2)
    # logging.info("Finished! :)")
