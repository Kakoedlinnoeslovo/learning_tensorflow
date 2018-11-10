__author__ = "kakoedlinnoeslovo"


from Logger import Logger
import numpy as np
import tensorflow as tf
import logging
Logger()
from model.CNN_utils import ConvRelu, max_2x2pooling, ConvReluBN, max_2x1pool, dropout

class CNN:
    def __init__(self, input_tensor, is_training):
        self.model = None
        self._build_network(input_tensor, is_training)


    def _build_network(self, input_tensor, is_training = False):
        logging.info("Start building CNN")
        logging.info("Input tensor shape: {}".format(input_tensor.get_shape()))
        net = tf.transpose(input_tensor, perm = [0, 2, 3, 1])
        net = tf.add(net, (-128.0))
        net = tf.multiply(net, (1/128.0))

        net = ConvRelu(net, 64, (3, 3), "conv_1")
        net = max_2x2pooling(net, name = "conv_pool1")

        logging.info("net tensor shape: {}".format(net.get_shape()))

        net = ConvRelu(net, 128, (3, 3), "conv_2")
        net = max_2x2pooling(net, name="conv_pool2")

        logging.info("net tensor shape: {}".format(net.get_shape()))

        net = ConvReluBN(net, 256, (3,3), 'conv3', is_training)
        net = ConvRelu(net, 256, (3,3), 'conv4')
        net = max_2x1pool(net, "conv_pool3")

        logging.info("net tensor shape: {}".format(net.get_shape()))

        net = ConvReluBN(net, 512, (3, 3), 'conv5', is_training, padding_type="VALID")
        net = ConvRelu(net, 512, (3, 3), 'conv6')
        net = max_2x1pool(net, "conv_pool4")

        logging.info("net tensor shape: {}".format(net.get_shape()))


        net = ConvReluBN(net, 512, (2,2), 'conv_conv7', is_training)
        net = dropout(net, is_training = is_training)

        print("CNN outdim before squeeze: {}".format(net.get_shape())) #1x32x100 -> 24x512

        net = tf.squeeze(net, axis = 1)

        print("CNN outdim: {}".format(net.get_shape()))

        self.model = net


    def get_output(self):
        return self.model


if __name__ == "__main__":
    img_data = tf.placeholder(dtype=tf.float32, shape=(None, 1, 32, 32), name="input_data")
    cnn = CNN(img_data, is_training=True)
    x = cnn.get_output()
