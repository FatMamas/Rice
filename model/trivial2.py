def build_network_trivial2(config, input_var=None):

    from lasagne.layers import InputLayer, DenseLayer
    from lasagne.nonlinearities import rectify, softmax
    import lasagne

    network = InputLayer(shape=(None, config.img_colors, config.img_size, config.img_size), input_var=input_var)
    network = DenseLayer(network, num_units=256, nonlinearity=softmax)
    network = DenseLayer(network, num_units=10, nonlinearity=softmax)

    return network, 'trivial2_tmp'
