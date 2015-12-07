def build_network_tomas2(config, input_var=None):
    # epoch 131 -> 76.28
    # epoch 171 -> 79.5654
    # epoch 179 -> 80.004
    # epoch 198 -> 81.03
    # epoch 259 -> 82.06
    # epoch 373 -> 83.789

    from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer, DenseLayer, ReshapeLayer
    from lasagne.nonlinearities import rectify, softmax
    try:
        from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    except ImportError:
        from lasagne.layers import Conv2DLayer as ConvLayer
        
    try:
        from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayer
    except ImportError:
        from lasagne.layers import Pool2DLayer as PoolLayer

    import lasagne

    network = InputLayer(shape=(None, config.img_colors, config.img_size, config.img_size), input_var=input_var)

    network = ConvLayer(network, num_filters=128, filter_size=5, nonlinearity=rectify)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    network = DropoutLayer(network, p=0.25)

    network = ConvLayer(network, num_filters=512, filter_size=3, nonlinearity=rectify)
    network = ConvLayer(network, num_filters=256, filter_size=3, nonlinearity=rectify)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    network = DropoutLayer(network, p=0.5)

    network = ConvLayer(network, num_filters=128, filter_size=3, nonlinearity=rectify)
    network = ConvLayer(network, num_filters=32, filter_size=2, nonlinearity=rectify)
    network = PoolLayer(network, pool_size=2, ignore_border=True)

    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=rectify)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=10, nonlinearity=softmax)

    return network, 'tomas2'
