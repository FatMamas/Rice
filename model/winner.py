def build_network_winner(config, input_var=None):
    """"input- 100C3-MP2- 200C2-MP2- 300C2-MP2- 400C2-MP2- 500C2-output"""

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
    print("I1", network.output_shape)

    network = ConvLayer(network, num_filters=250, filter_size=5, nonlinearity=rectify)
    print("C1", network.output_shape)
    network = ConvLayer(network, num_filters=100, filter_size=3, nonlinearity=rectify)
    print("C2", network.output_shape)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    print("P1", network.output_shape)

    network = ConvLayer(network, num_filters=250, filter_size=2, nonlinearity=rectify)
    print("C3", network.output_shape)
    network = ConvLayer(network, num_filters=250, filter_size=2, nonlinearity=rectify)
    print("C4", network.output_shape)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    print("P2", network.output_shape)

    network = ConvLayer(network, num_filters=250, filter_size=2, nonlinearity=rectify)
    print("C5", network.output_shape)
    network = ConvLayer(network, num_filters=100, filter_size=2, nonlinearity=rectify)
    print("C6", network.output_shape)
    network = PoolLayer(network, pool_size=2, ignore_border=True)
    print("P3", network.output_shape)

    # network = ConvLayer(network, num_filters=500, filter_size=(2, 2), nonlinearity=rectify)
    # print("C5", network.output_shape)

    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=10, nonlinearity=softmax)
    print("D1", network.output_shape)

    return network, 'winner_big_mean'
