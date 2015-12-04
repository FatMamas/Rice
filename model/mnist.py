def build_network(config, input_var=None):
    import lasagne

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

    return network, 'mnist'
