import tensorflow as tf
import math


def var_random(name, shape,
               regularizable = False):

    assert isinstance(shape, tuple), "shape should be tuple"

    filters_in = shape[2]
    filters_out = shape[3]

    variation = 2/(filters_in + filters_out)
    dev = math.sqrt(variation)
    v = tf.get_variable(name, initializer=tf.random_normal(shape = shape,
                                                     mean=0,
                                                     stddev=dev))

    if regularizable:
        with tf.name_scope(name + "/Regularizer/"):
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(v))
    return v


def ConvRelu(input, num_filters, filter_size, name):
    num_filters_from = input.get_shape().as_list()[3]
    with tf.variable_scope(name):
        conv_W = var_random("W", tuple(filter_size) + (num_filters_from, num_filters),
                            regularizable=True)

        after_conv = tf.nn.conv2d(input = input, filter=conv_W, strides=(1,1,1,1), padding= "SAME")

        return tf.nn.relu(after_conv)


def max_2x2pooling(incoming, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize = (1,2,2,1), strides=(1,2,2,1), padding = "VALID")


def batch_norm(incoming, is_training):
    return tf.contrib.layers.batch_norm(incoming,
                                        is_training = is_training,
                                        scale = True,
                                        decay = 0.99)

def ConvReluBN(incoming, num_filters, filter_size, name, is_training, padding_type = "SAME"):
    num_filters_from = incoming.get_shape().as_list()[3]
    with tf.variable_scope(name):
        #shape var_random, for example: 3,3, 128, 256
        conv_W = var_random(name = "W", shape = tuple(filter_size) + (num_filters_from, num_filters),
                            regularizable=True)
        after_conv = tf.nn.conv2d(input=incoming, filter=conv_W,
                                  strides=(1,1,1,1), padding=padding_type)
        after_bn = batch_norm(after_conv, is_training = is_training)
        return tf.nn.relu(after_bn)


def max_2x1pool(incoming, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize = (1,2,1,1), strides = (1,2,1,1), padding="VALID")


def dropout(incoming, is_training, keep_prob = 0.5):
    return tf.contrib.layers.dropout(incoming, keep_prob = keep_prob, is_training = is_training)