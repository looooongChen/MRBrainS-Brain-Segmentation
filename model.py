import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
# DEFAULT_DTYPE = tf.float32
_SCALE = 0.01


def conv_batch_norm(inputs, training, axis=3):

    return tf.layers.batch_normalization(inputs=inputs, axis=axis,
                                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                         scale=True, training=training, fused=True)


def avg_pool(inputs, k, stride, padding='VALID'):

    inputs = tf.nn.avg_pool(inputs,
                            ksize=[1, k, k, 1],
                            strides=[1, stride, stride, 1],
                            padding=padding)
    return inputs


def resize_bilinear(inputs, size):

    return tf.image.resize_bilinear(inputs, size=size, align_corners=False)


def conv2d_same(inputs, filters, kernel_size, dilation_rate, strides=1, padding='SAME', use_bias=False):

    return tf.layers.conv2d(inputs=inputs, filters=filters,
                            kernel_size=kernel_size, strides=strides, dilation_rate=(dilation_rate, dilation_rate),
                            padding=padding, use_bias=use_bias, activation=None,
                            kernel_initializer=tf.variance_scaling_initializer(scale=_SCALE))


def init_block(inputs, features, kernel_size, reduce=2):

    inputs = conv2d_same(inputs=inputs, filters=features, kernel_size=kernel_size,
                         dilation_rate=1, use_bias=True)
    # inputs = tf.nn.relu(inputs)
    # inputs = conv2d_same(inputs=inputs, filters=filters, kernel_size=kernel_size,
    #                      dilation_rate=1)
    # inputs = tf.nn.relu(inputs)
    # inputs = conv2d_same(inputs=inputs, filters=filters, kernel_size=kernel_size,
    #                      dilation_rate=1)
    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[reduce, reduce], strides=reduce)
    return inputs


def bottleneck(inputs, filters, factor, training, dilation_rate=1):

    shortcut = inputs
    inputs = conv_batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    # 1x1 conv
    res = conv2d_same(inputs=inputs, filters=filters, kernel_size=1, dilation_rate=1, strides=1)
    res = conv_batch_norm(res, training)
    res = tf.nn.relu(res)
    # 3x3 conv
    res = conv2d_same(inputs=res, filters=filters, kernel_size=3, dilation_rate=dilation_rate, strides=1)
    res = conv_batch_norm(res, training)
    res = tf.nn.relu(res)
    # 1x1 conv
    res = conv2d_same(inputs=res, filters=filters * factor, kernel_size=1, dilation_rate=1, strides=1)

    # projection shortcut
    if res.shape[3] != inputs.shape[3]:
        shortcut = conv2d_same(inputs=inputs, filters=filters * factor, kernel_size=1, dilation_rate=1, strides=1)

    return shortcut + res


def conv_block(inputs, filters, factor, bottleneck_num, dilation_rate, training, stream=1):

    for i in range(0, bottleneck_num-1):
        with tf.variable_scope('bottleneck{}'.format(i + 1)):
            inputs = bottleneck(inputs, filters*stream, factor, training, dilation_rate)
    with tf.variable_scope('bottleneck{}'.format(bottleneck_num)):
        inputs = bottleneck(inputs, filters*stream, factor/stream, training, dilation_rate)

    return inputs


def pp_block(inputs, scale, filter_num):
    shape = tf.shape(inputs)[1:3]
    inputs = avg_pool(inputs, scale, scale)
    inputs = conv2d_same(inputs, filter_num, 1, 1, 1)
    inputs = tf.nn.relu(inputs)
    inputs = resize_bilinear(inputs, shape)

    return inputs


def make_mixNet_plain(inputs, training, paras):

    with tf.variable_scope('initial_module'):
        initial_outputs = init_block(inputs, paras['filter_num'], 5,
                                     reduce=paras['subsample'])
    # inputs are not activated (without activation function)

    with tf.variable_scope('level1'):
        l1_outputs = conv_block(inputs=initial_outputs,
                                filters=paras['filter_num'], factor=self.factor,
                                bottleneck_num=paras['block_num'][0],
                                dilation_rate=paras['block_dilation'][0],
                                training=training)

    with tf.variable_scope('level2'):
        l2_outputs = conv_block(inputs=l1_outputs, filters=paras['filter_num'],
                                factor=paras['block_factor'],
                                bottleneck_num=paras['block_num'][1],
                                dilation_rate=paras['block_dilation'][1],
                                training=training)

    with tf.variable_scope('level3'):
        l3_outputs = conv_block(inputs=l2_outputs, filters=paras['filter_num'],
                                factor=paras['block_factor'],
                                bottleneck_num=paras['block_num'][2],
                                dilation_rate=paras['block_dilation'][2],
                                training=training)

    with tf.variable_scope('level4'):
        l4_outputs = conv_block(inputs=l3_outputs, filters=paras['filter_num'],
                                factor=paras['block_factor'],
                                bottleneck_num=paras['block_num'][3],
                                dilation_rate=paras['block_dilation'][3],
                                training=training)

    with tf.variable_scope('level5'):
        l5_outputs = conv_block(inputs=l4_outputs, filters=paras['filter_num'],
                                factor=paras['block_factor'],
                                bottleneck_num=paras['block_num'][4],
                                dilation_rate=paras['block_dilation'][4],
                                training=training)

    with tf.variable_scope('merge'):
        outputs = tf.concat([l1_outputs,
                             l2_outputs, l3_outputs,
                             l4_outputs, l5_outputs], -1)
        outputs = tf.nn.relu(outputs)
        if paras['subsample'] != 1:
            outputs = resize_bilinear(outputs, inputs.shape[1:3])

    if paras['with_pp']:
        with tf.variable_scope('pp_module'):
            pp1 = pp_block(outputs, 120, paras['filter_num'])
            pp2 = pp_block(outputs, 60, paras['filter_num'])
            pp3 = pp_block(outputs, 40, paras['filter_num'])
            pp4 = pp_block(outputs, 20, paras['filter_num'])

    with tf.variable_scope('outputs_module'):
        if paras['with_pp']:
            outputs = tf.concat([outputs, pp1, pp2, pp3, pp4], -1)
        outputs = conv2d_same(outputs, paras['class_num'], 3, 1, 1, use_bias=True)

    return outputs


def make_mixNet_combi(inputs, training, paras):

    initial_inputs = []
    initial_outputs = []
    l1_outputs = []
    l2_outputs = []
    l3_outputs = []
    l4_outputs = []
    l5_outputs = []
    merge = []

    for i in range(paras['stream_num']):
        initial_inputs.append(tf.expand_dims(inputs[:, :, :, i], axis=3))

    with tf.variable_scope('initial_module'):
        for i in range(paras['stream_num']):
            initial_outputs.append(init_block(initial_inputs[i],
                                              paras['filter_num'], 5,
                                              reduce=paras['subsample']))
    # inputs are not activated (without activation function)

    with tf.variable_scope('level1'):
        for i in range(paras['stream_num']):
            with tf.variable_scope('stream'+str(i)):
                l1_outputs.append(conv_block(inputs=initial_outputs[i],
                                             filters=paras['filter_num'],
                                             factor=paras['block_factor'],
                                             bottleneck_num=paras['block_num'][0],
                                             dilation_rate=paras['block_dilation'][0],
                                             training=training))

    with tf.variable_scope('level2'):
        for i in range(paras['stream_num']):
            with tf.variable_scope('stream'+str(i)):
                l2_outputs.append(conv_block(inputs=l1_outputs[i],
                                             filters=paras['filter_num'],
                                             factor=paras['block_factor'],
                                             bottleneck_num=paras['block_num'][1],
                                             dilation_rate=paras['block_dilation'][1],
                                             training=training))

    with tf.variable_scope('level3'):
        for i in range(paras['stream_num']):
            with tf.variable_scope('stream'+str(i)):
                l3_outputs.append(conv_block(inputs=l2_outputs[i],
                                             filters=paras['filter_num'],
                                             factor=paras['block_factor'],
                                             bottleneck_num=paras['block_num'][2],
                                             dilation_rate=paras['block_dilation'][2],
                                             training=training))

    with tf.variable_scope('level4'):
        for i in range(paras['stream_num']):
            with tf.variable_scope('stream'+str(i)):
                l4_outputs.append(conv_block(inputs=l3_outputs[i],
                                             filters=paras['filter_num'],
                                             factor=paras['block_factor'],
                                             bottleneck_num=paras['block_num'][3],
                                             dilation_rate=paras['block_dilation'][3],
                                             training=training))
    with tf.variable_scope('level5'):
        for i in range(paras['stream_num']):
            with tf.variable_scope('stream'+str(i)):
                l5_outputs.append(conv_block(inputs=l4_outputs[i],
                                             filters=paras['filter_num'],
                                             factor=paras['block_factor'],
                                             bottleneck_num=paras['block_num'][4],
                                             dilation_rate=paras['block_dilation'][4],
                                             training=training))

    with tf.variable_scope('merge'):
        for i in range(paras['stream_num']):
            merge.append(tf.concat([initial_outputs[i], l1_outputs[i], l2_outputs[i],
                                    l3_outputs[i], l4_outputs[i], l5_outputs[i]], -1))
        outputs = tf.concat(merge, -1)
        outputs = tf.nn.relu(outputs)
        if paras['subsample'] != 1:
            outputs = resize_bilinear(outputs, inputs.shape[1:3])

    if paras['with_pp']:
        with tf.variable_scope('pp_module'):
            pp1 = pp_block(outputs, 120, paras['filter_num'])
            pp2 = pp_block(outputs, 60, paras['filter_num'])
            pp3 = pp_block(outputs, 40, paras['filter_num'])
            pp4 = pp_block(outputs, 20, paras['filter_num'])

    with tf.variable_scope('outputs_module'):
        if paras['with_pp']:
            outputs = tf.concat([outputs, pp1, pp2, pp3, pp4], -1)
        outputs = conv2d_same(outputs, paras['class_num'], 3, 1, 1, use_bias=True)

    return outputs


def make_mixNet_mix(inputs, training, paras):

    initial_inputs = []
    initial_outputs = []
    l2_outputs = []
    l4_outputs = []
    merge = []

    for i in range(paras['stream_num']):
        initial_inputs.append(tf.expand_dims(inputs[:, :, :, i], axis=3))

    with tf.variable_scope('initial_module'):
        for i in range(paras['stream_num']):
            initial_outputs.append(init_block(initial_inputs[i],
                                              paras['filter_num'], 5,
                                              reduce=paras['subsample']))
    # inputs are not activated (without activation function)

    with tf.variable_scope('level1'):
        l1_outputs = conv_block(inputs=tf.concat(initial_outputs, -1),
                                filters=paras['filter_num'],
                                factor=paras['block_factor'],
                                bottleneck_num=paras['block_num'][0],
                                dilation_rate=paras['block_dilation'][0],
                                training=training)

    with tf.variable_scope('level2'):
        for i in range(paras['stream_num']):
            l2_outputs.append(inputs=tf.concat([l1_outputs, initial_outputs[i]]),
                              filters=paras['filter_num'],
                              factor=paras['block_factor'],
                              bottleneck_num=paras['block_num'][1],
                              dilation_rate=paras['block_dilation'][1],
                              training=training)

    with tf.variable_scope('level3'):
        l3_outputs = conv_block(inputs=tf.concat(initia2_outputs, -1),
                                filters=paras['filter_num'],
                                factor=paras['block_factor'],
                                bottleneck_num=paras['block_num'][2],
                                dilation_rate=paras['block_dilation'][2],
                                training=training)

    with tf.variable_scope('level4'):
        for i in range(paras['stream_num']):
            l4_outputs.append(inputs=tf.concat([l3_outputs, l2_outputs[i]]),
                              filters=paras['filter_num'],
                              factor=paras['block_factor'],
                              bottleneck_num=paras['block_num'][3],
                              dilation_rate=paras['block_dilation'][3],
                              training=training)

    with tf.variable_scope('level5'):
        l5_outputs = conv_block(inputs=tf.concat(initia4_outputs, -1),
                                filters=paras['filter_num'],
                                factor=paras['block_factor'],
                                bottleneck_num=paras['block_num'][4],
                                dilation_rate=paras['block_dilation'][4],
                                training=training)

    with tf.variable_scope('merge'):
        for i in range(paras['stream_num']):
            merge.append(tf.concat([initial_outputs[i], l2_outputs[i],
                                    l4_outputs[i]], -1))
        outputs = tf.concat(merge+[l1_outputs, l3_outputs, l5_outputs], -1)
        outputs = tf.nn.relu(outputs)
        if paras['subsample'] != 1:
            outputs = resize_bilinear(outputs, inputs.shape[1:3])

    if paras['with_pp']:
        with tf.variable_scope('pp_module'):
            pp1 = pp_block(outputs, 120, paras['filter_num'])
            pp2 = pp_block(outputs, 60, paras['filter_num'])
            pp3 = pp_block(outputs, 40, paras['filter_num'])
            pp4 = pp_block(outputs, 20, paras['filter_num'])

    with tf.variable_scope('outputs_module'):
        if paras['with_pp']:
            outputs = tf.concat([outputs, pp1, pp2, pp3, pp4], -1)
        outputs = conv2d_same(outputs, paras['class_num'], 3, 1, 1, use_bias=True)

    return outputs


def make_model(inputs, training, config):
    if config['type'] == 'plain':
        return make_mixNet_plain(inputs, training, config)
    elif config['type'] == 'mix':
        return make_mixNet_mix(inputs, training, config)
    elif config['type'] == 'combi':
        return make_mixNet_combi(inputs, training, config)
    else:
        print("Unknow net architecture!")


