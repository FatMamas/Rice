def build_cifar_network(config, input_var=None):
    from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer
    try:
        from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    except ImportError:
        from lasagne.layers import Conv2DLayer as ConvLayer
        
    try:
        from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
    except ImportError:
        from lasagne.layers import Pool2DLayer as PoolLayer
    
    # what I changed
    # imported Pool2DDNNLayer, for that I needed to set ignore_border=True
    # and that in turn needed to change the last PoolLayer to size=4 from 8 - because the ccp5,6 had size (7, 7), and 7/8 would be 0, but 7/4 is 1, so that's what we want
    # btw, I think we should put a 'DenseLayer' somewhere in the end instead of the Pool/Flatten Layer

    net = {}
    net['input'] = InputLayer(shape=(None, config.img_colors, config.img_size, config.img_size), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'], num_filters=192, filter_size=5, pad=2)
    net['cccp1'] = ConvLayer(net['conv1'], num_filters=160, filter_size=1)
    net['cccp2'] = ConvLayer(net['cccp1'], num_filters=96, filter_size=1)
    net['pool1'] = PoolLayer(net['cccp2'], pool_size=3, stride=2, mode='max', ignore_border=True)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'], num_filters=192, filter_size=5, pad=2)
    net['cccp3'] = ConvLayer(net['conv2'], num_filters=192, filter_size=1)
    net['cccp4'] = ConvLayer(net['cccp3'], num_filters=192, filter_size=1)
    net['pool2'] = PoolLayer(net['cccp4'], pool_size=3, stride=2, mode='average_exc_pad', ignore_border=True)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'], num_filters=192, filter_size=3, pad=1)
    net['cccp5'] = ConvLayer(net['conv3'], num_filters=192, filter_size=1)
    net['cccp6'] = ConvLayer(net['cccp5'], num_filters=10, filter_size=1)
    net['pool3'] = PoolLayer(net['cccp6'], pool_size=4, mode='average_exc_pad', ignore_border=True)
    net['output'] = FlattenLayer(net['pool3'])

    return net['output'], 'official'
    #return net