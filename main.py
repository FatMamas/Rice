import logging
import sys
import argparse
import os
import random

import pickle
import numpy as np

DATA_DIR = "data"
DATA_FILENAME = "cifar-10-python.tar.gz"
DATA_PATH = DATA_DIR + "/" + DATA_FILENAME

# BATCH_NUMBER = 6 # CIFAR10
BATCH_NUMBER = 2 # CIFAR100
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
    part_labels = np.append(np.empty(shape=0, dtype=np.uint8), file_data[b'fine_labels'])
    return part_data.reshape((-1, config.img_colors, config.img_size, config.img_size)).astype(np.float32)/128.0 - 1.0, part_labels.astype(np.int8)


def load_dataset():
    all_data = np.empty(shape=(0, config.img_colors, config.img_size, config.img_size), dtype=np.float32)
    all_labels = np.empty(shape=0, dtype=np.int8)

    for i in range(1, BATCH_NUMBER):
        chunk_data, chunk_labels = load_single_dataset(BATCH_PATH.format(i))
        all_data = np.append(all_data, chunk_data)
        all_labels = np.append(all_labels, chunk_labels)

    return all_data.reshape((-1, config.img_colors, config.img_size, config.img_size)), all_labels.astype(np.int8)


def save_network(filename, network_p, net_name_p, input_var_p, target_var_p, prediction_p, loss_p, params_p, updates_p,
                 test_prediction_p, test_loss_p, predict_fn_p, test_acc_p, train_fn_p, val_fn_p):
    data = {
        'network': network_p,
        'net_name': net_name_p,
        'input_var': input_var_p,
        'target_var': target_var_p,
        'prediction': prediction_p,
        'loss': loss_p,
        'params': params_p,
        'updates': updates_p,
        'test_prediction': test_prediction_p,
        'test_loss': test_loss_p,
        'predict_fn': predict_fn_p,
        'test_acc': test_acc_p,
        'train_fn': train_fn_p,
        'val_fn': val_fn_p
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_network(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data['network'], data['net_name'], data['input_var'], data['target_var'], data['prediction'], data['loss'],\
           data['params'], data['updates'], data['test_prediction'], data['test_loss'], data['predict_fn'],\
           data['test_acc'], data['train_fn'], data['val_fn']


def save_network_old(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lasagne.layers.get_all_param_values(net), f)


def load_network_old(filename, net):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(net, data)
    return net


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        # yield inputs[1:50], targets[1:50] # TODO: do not use this! never!


def recall(recall_fn, patterns):
    """NN recall. Recalls a single image or an array of images. Images are supposed to be NumPy arrays."""

    if patterns.ndim == 4:
        return recall_fn(patterns)
    elif patterns.ndim == 3:
        return recall_fn([patterns])[0]
    else:
        raise ValueError('Unexpected dimension of the patterns parameter')


def test_random(recall_fn, patterns, expectations, n=10):
    for i in range(n):
        idx = random.randint(1, len(patterns)-2)
        res = recall(recall_fn, patterns[idx])
        exp = expectations[idx]
        logging.info('Test image #%d:\t recall: %d\t expected: %d', idx, res, exp)


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
    parser.add_argument("-b", "--minibatch", default=128, type=int, help="Size of the minibatch")
    parser.add_argument("-m", "--mode", default="FAST_RUN", help="Theano run mode")
    parser.add_argument("-f", "--floatX", default="float32", help="Theano floatX mode")
    parser.add_argument("-l", "--log", default="log", help="Log directory")
    parser.add_argument("-o", "--output", default="trained", help="Trained model output directory")
    parser.add_argument("-r", "--restore", default=None, help="Path to the saved model to be continued from")
    parser.add_argument("-i", "--iter", default=0, help="Number of first epoch. Use together with --restore in order to have beautiful logs")

    global config
    config = parser.parse_args()
    config.device = config.device[0]
    config.img_colors = 3
    config.img_size = 32
    config.iter = int(config.iter)

    logging.info("Setting environmental variables for Theano")
    os.environ["THEANO_FLAGS"] = "mode={},device={},floatX={},nvcc.fastmath=True".format(config.mode, config.device, config.floatX)

    logging.info("Importing Theano and Lasagne")
    import theano
    import theano.tensor as T
    import lasagne

    logging.info("Setting the recursion max limit")
    sys.setrecursionlimit(config.minibatch + 10000)

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

    if config.restore:
        logging.info("Restoring the model and all variables from '%s'", config.restore)
        network, net_name, input_var, target_var, prediction, loss, params, updates, test_prediction, test_loss, predict_fn, test_acc, train_fn, val_fn = load_network(config.restore)
    else:   # build new model
        logging.info("Creating Theano input and target variables")
        input_var = T.tensor4('inputs', dtype='float32')
        target_var = T.ivector('targets')

        logging.info("Importing the network module")    # we need to import this AFTER Theano and Lasagne
        # from model.mnist import build_network as build_network
        # from model.official import build_cifar_network as build_network
        # from model.conv3 import build_network_3cc as build_network
        # from model.conv4 import build_network_4cc as build_network
        # from model.winner import build_network_winner as build_network
        # from model.tomas import build_network_tomas as build_network
        # from model.tomas2 import build_network_tomas2 as build_network
        # from model.tomas2_1 import build_network_tomas2_1 as build_network
        # from model.tomas3 import build_network_tomas3 as build_network
        # from model.trivial import build_network_trivial as build_network
        # from model.trivial2 import build_network_trivial2 as build_network
        # from model.conv1 import build_network_1cc as build_network
        from model.cifar100 import build_network_cifar100 as build_network

        logging.info("Building the network")
        network, net_name = build_network(config, input_var)
        logging.info("Network '%s' successfully built", net_name)

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

        logging.info("Creating the prediction expression")
        predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

        logging.info("Creating the test accuracy expression")
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        logging.info("Compiling the train function (Theano)")
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        logging.info("Compiling the validation function (Theano)")
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        logging.info("Creating log directory")
        os.makedirs(config.log, exist_ok=True)

        logging.info("Creating output directory")
        os.makedirs(config.output, exist_ok=True)

    best_acc = -1
    last_save = 0

    write_mode = 'a' if config.restore else 'w'     # continue in previous log if the model is restored
    with open('{}/{}.csv'.format(config.log, net_name), write_mode) as log_f:
        if not config.restore:
            log_f.write("epoch;trainloss;valloss;valacc\n")

        logging.info("Starting the training loop")
        for epoch in range(config.iter, config.trainepochs):
            logging.info("Epoch #%d", epoch)

            logging.info("Passing over the training data")
            train_err = 0
            train_batches = 0
            for batch_id, (inputs, targets) in enumerate(iterate_minibatches(train_data, train_labels, config.minibatch, shuffle=True)):
                # logging.info("Batch %d in epoch #%d", batch_id, epoch)
                err = train_fn(inputs, targets)
                logging.info("train err: " + str(err))
                train_err += err
                train_batches += 1

            logging.info("Passing over the validation data")
            val_err = 0
            val_acc = 0
            val_batches = 0
            for inputs, targets in iterate_minibatches(test_data, test_labels, config.minibatch, shuffle=False):
                err, acc = val_fn(inputs, targets)
                logging.info("test err: " + str(err) + ", acc: " + str(acc))
                val_err += err
                val_acc += acc
                val_batches += 1

            logging.info(train_err)
            logging.info(train_batches)
            logging.info(val_err)
            logging.info(val_batches)
            logging.info("Training loss:\t\t%.10f", train_err / train_batches)
            logging.info("Validation loss:\t\t%.10f", val_err / val_batches)
            logging.info("Validation accuracy:\t%.20f%%", val_acc / val_batches * 100)

            # print nasty progress info
            log_f.write("{};{};{};{}\n".format(epoch, train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100))
            log_f.flush()

            # save better model if it's better then the previous best one
            if val_acc / val_batches > best_acc:
                best_acc = val_acc / val_batches
                last_save = epoch
                network_filename = "{}/network_{}_{}.dat".format(config.output, net_name, epoch)
                logging.info("Saving the model into '%s'", network_filename)
                save_network(network_filename, network, net_name, input_var, target_var, prediction, loss, params,
                             updates, test_prediction, test_loss, predict_fn, test_acc, train_fn, val_fn)

    logging.info("Training finished")

    network_filename = "{}/network_{}_{}.dat".format(config.output, net_name, last_save)
    logging.info("Reading it just to be sure: {}".format(network_filename))
    network_n, net_name_n, input_var_n, target_var_n, prediction_n, loss_n, params_n, updates_n, test_prediction_n, test_loss_n, predict_fn_n, test_acc_n, train_fn_n, val_fn_n = load_network(network_filename)

    logging.info("Printing 20 random classifications")
    test_random(predict_fn_n, test_data, test_labels, n=20)

    logging.info("Finished! Unbelievable, right?")
