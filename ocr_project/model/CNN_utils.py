import tensorflow as tf
import math


def var_random(name, shape,
               filters_in,
               filters_out,
               regularizable = False):

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
                            filters_in = num_filters_from,
                            filters_out = num_filters, regularizable=True)

        conv = tf.nn.conv2d(input = input, filter=conv_W, strides=(1,1,1,1), padding= "SAME")

        return tf.nn.relu(conv)


def max_2x2pooling(incoming, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize = (1,2,2,1), strides=(1,2,2,1), padding = "VALID")




