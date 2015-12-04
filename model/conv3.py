def build_network_3cc(config, input_var=None):
    from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    try:
        from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    except ImportError:
        from lasagne.layers import Conv2DLayer as ConvLayer

    net = InputLayer(shape=(None, config.img_colors, config.img_size, config.img_size), input_var=input_var)

    net = ConvLayer(net, num_filters=160, filter_size=2)
    net = PoolLayer(net, pool_size=3, stride=2, mode='max', ignore_border=False)
    net = DropoutLayer(net, p=0.5)

    net = ConvLayer(net, num_filters=192, filter_size=1)
    net = PoolLayer(net, pool_size=3, stride=2, mode='average_exc_pad', ignore_border=False)
    net = DropoutLayer(net, p=0.5)

    net = ConvLayer(net, num_filters=10, filter_size=1)
    net = PoolLayer(net, pool_size=8, mode='average_exc_pad', ignore_border=False)
    net = FlattenLayer(net)

    return net, '3cc'