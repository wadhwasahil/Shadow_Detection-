import tensorflow as tf


def default_conv(input, num_filters):
    b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
    conv = tf.nn.bias_add(tf.layers.conv2d(input, filters=num_filters, kernel_size=(5, 5),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                                           strides=(2, 2), padding="VALID"), b, name="conv")
    return conv


class Generator(object):
    def __init__(self, w=0.1):
        self.X = tf.placeholder(tf.float32, [None, 256, 256, 3], name="X_train")

        with tf.name_scope("conv64"):
            conv = default_conv(self.X, 64)

        with tf.name_scope("conv_RB128"):
            r = tf.nn.leaky_relu(conv, name="leaky_relu")
            conv = default_conv(r, 128)
            batch_norm = tf.layers.batch_normalization(conv, training=True)

        with tf.name_scope("conv_RB256"):
            r = tf.nn.leaky_relu(batch_norm, name="leaky_relu")
            conv = default_conv(r, 256)
            batch_norm = tf.layers.batch_normalization(conv, training=True)

        with tf.name_scope("conv_RB512_1"):
            r = tf.nn.leaky_relu(batch_norm, name="leaky_relu")
            conv = default_conv(r, 512)
            batch_norm = tf.layers.batch_normalization(conv, training=True)

        with tf.name_scope("conv_RB512_2"):
            r = tf.nn.leaky_relu(batch_norm, name="leaky_relu")
            conv = default_conv(r, 512)
            batch_norm = tf.layers.batch_normalization(conv, training=True)

        with tf.name_scope("conv_RB512_3"):
            r = tf.nn.leaky_relu(batch_norm, name="leaky_relu")
            conv = default_conv(r, 512)
            batch_norm = tf.layers.batch_normalization(conv, training=True)

        with tf.name_scope("conv_RB512_4"):
            r = tf.nn.leaky_relu(batch_norm, name="leaky_relu")
            conv = default_conv(r, 512)
            batch_norm = tf.layers.batch_normalization(conv, training=True)

        with tf.name_scope("conv_RB512_5"):
            r = tf.nn.leaky_relu(batch_norm, name="leaky_relu")
            conv = default_conv(r, 512)
            batch_norm = tf.layers.batch_normalization(conv, training=True)
