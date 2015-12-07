def build_network_4cc(config, input_var=None):
    from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer, DenseLayer, ReshapeLayer
    from lasagne.nonlinearities import rectify, softmax
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
    # print("I1", network.output_shape)

    network = ConvLayer(network, num_filters=512, filter_size=(5, 5), nonlinearity=rectify)
    # print("C1", network.output_shape)
    network = PoolLayer(network, pool_size=(2, 2), mode='max', ignore_border=True)
    # print("P1", network.output_shape)

    network = ConvLayer(network, num_filters=256, filter_size=(4, 4), nonlinearity=rectify)
    # print("C2", network.output_shape)
    network = PoolLayer(network, pool_size=(2, 2), mode='max', ignore_border=True)
    # print("P2", network.output_shape)
    network = DropoutLayer(network, p=0.5)

    network = ConvLayer(network, num_filters=64, filter_size=(3, 3), nonlinearity=rectify)
    # print("C3", network.output_shape)
    network = PoolLayer(network, pool_size=(2, 2), mode='max', ignore_border=True)
    # print("P3", network.output_shape)

    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=rectify)
    # print(network.output_shape)
    network = DenseLayer(network, num_units=128, nonlinearity=rectify)
    # print(network.output_shape)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=10, nonlinearity=softmax)
    # print(network.output_shape)

    return network, 'conv3_drop'
