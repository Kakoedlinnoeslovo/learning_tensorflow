__author__ = "kakoedlinnoeslovo"


from Logger import Logger
import numpy as np
import tensorflow as tf
import logging
Logger()
from model.CNN_utils import ConvRelu

class CNN:
    def __init__(self):
        pass

    def _build_network(self, input_tensor, is_training = False):
        logging.info("Start building CNN")
        logging.info("Input tensor shape: {}".format(input_tensor.get_shape()))
        net = tf.transpose(input_tensor, perm = [0, 2, 3, 1])
        net = tf.add(net, (-128.0))
        net = tf.multiply(net, (1/128.0))

        net = ConvRelu(net, 64, (3, 3), "conv_1")




if __name__ == "__main__":
    cnn = CNN()
    img_data = tf.placeholder(dtype = tf.float32, shape = (None, 1, 32, None), name = "input_data")
    cnn._build_network(img_data)
