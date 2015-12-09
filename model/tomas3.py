def build_network_tomas3(config, input_var=None):
    from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
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

    network = ConvLayer(network, num_filters=256, stride=2, untie_biases=True, filter_size=3, nonlinearity=rectify)
    network = ConvLayer(network, num_filters=128, stride=2, filter_size=3, nonlinearity=rectify)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    network = DropoutLayer(network, p=0.25)

    network = ConvLayer(network, num_filters=512, filter_size=3, nonlinearity=rectify)
    network = ConvLayer(network, num_filters=256, filter_size=3, nonlinearity=rectify)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    network = DropoutLayer(network, p=0.5)

    network = ConvLayer(network, num_filters=256, filter_size=3, nonlinearity=rectify)
    network = ConvLayer(network, num_filters=128, filter_size=2, nonlinearity=rectify)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    network = DropoutLayer(network, p=0.125)

    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=512, nonlinearity=rectify)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=128, nonlinearity=rectify)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=10, nonlinearity=softmax)

    return network, 'tomas3'
