def build_network_1cc(config, input_var=None):

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
    network = ConvLayer(network, num_filters=256, filter_size=(5, 5), nonlinearity=rectify)
    network = PoolLayer(network, pool_size=(2, 2), mode='max', ignore_border=True)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=rectify)
    network = DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=10, nonlinearity=softmax)

    return network, 'conv1'
