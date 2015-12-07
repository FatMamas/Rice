def build_network(config, input_var=None):

    from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer, DenseLayer
    try:
        from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    except ImportError:
        from lasagne.layers import Conv2DLayer as ConvLayer
        
    try:
        from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
    except ImportError:
        from lasagne.layers import Pool2DLayer as PoolLayer

    import lasagne

    network = InputLayer(shape=(None, config.img_colors, config.img_size, config.img_size), input_var=input_var)

    # Convolutional layer with IMG_SIZE kernels of size 5x5.
    network = ConvLayer(
            network, num_filters=config.img_size, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = PoolLayer(network, pool_size=(2, 2))

    # Another convolution with IMG_SIZE 5x5 kernels
    network = ConvLayer(
            network, num_filters=config.img_size, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    # And another 2x2 pooling:
    network = PoolLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network, 'mnist'
