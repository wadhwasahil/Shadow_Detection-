import tensorflow as tf


def default_conv(input, num_filters):
    input_channels = input.get_shape().as_list()[-1]
    b = tf.get_variable('b' + str(num_filters), [num_filters], initializer=tf.constant_initializer(0))
    W = tf.get_variable('weights' + str(num_filters), [5, 5, input_channels, num_filters],
                        initializer=tf.truncated_normal_initializer(stddev=0.2))
    conv = tf.nn.bias_add(tf.nn.conv2d(input, filter=W, strides=[1, 2, 2, 1], padding="SAME"), b, name="conv")
    return conv


class Discriminator(object):
    def __init__(self, rgb_image, shadow_mask, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self.shadow_mask_expanded = tf.expand_dims(shadow_mask, axis=-1, name="shadow_mask_expanded")
            self.stack = tf.concat([rgb_image, self.shadow_mask_expanded], axis=-1)

            with tf.variable_scope("conv_R64"):
                conv1 = default_conv(self.stack, 64)
                r1 = tf.nn.leaky_relu(conv1, name="leaky_relu")

            with tf.variable_scope("conv_BR128"):
                conv2 = default_conv(r1, 128)
                batch_norm1 = tf.layers.batch_normalization(conv2, training=True)
                r2 = tf.nn.leaky_relu(batch_norm1, name="leaky_relu")

            with tf.variable_scope("conv_BR256"):
                conv3 = default_conv(r2, 256)
                batch_norm2 = tf.layers.batch_normalization(conv3, training=True)
                r3 = tf.nn.leaky_relu(batch_norm2, name="leaky_relu")

            with tf.variable_scope("conv_BR512"):
                conv4 = default_conv(r3, 512)
                batch_norm3 = tf.layers.batch_normalization(conv4, training=True)
                r4 = tf.nn.leaky_relu(batch_norm3, name="leaky_relu")
            shape = r4.get_shape()
            num_features = shape[1:].num_elements()
            with tf.variable_scope("fc"):
                reshape = tf.reshape(r4, [-1, num_features], name="reshape")
                W = tf.get_variable('weights', shape=[num_features, 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.2))
                b = tf.get_variable('biases', shape=[1], initializer=tf.constant_initializer(0.0))
                self.logits = tf.matmul(reshape, W) + b
                self.sigmoid = tf.nn.sigmoid(self.logits, name="sigmoid")
