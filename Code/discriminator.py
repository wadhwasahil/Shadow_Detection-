import tensorflow as tf

def default_conv(input, num_filters):
    b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
    conv = tf.nn.bias_add(tf.layers.conv2d(input, filters=num_filters, kernel_size=(5, 5),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                                           strides=(2, 2), padding="same"), b, name="conv")
    return conv

class Discriminator(object):
    def __init__(self):
        self.rgb_image = tf.placeholder(tf.float32, [None, 256, 256, 3], name="rg_image")
        self.shadow_mask = tf.placeholder(tf.float32, [None, 256, 256], name="shadow_mask")
        self.shadow_mask_expanded = tf.expand_dims(self.shadow_mask, axis=-1, name="shadow_mask_expanded")
        self.stack = tf.concat([self.rgb_image, self.shadow_mask_expanded], axis=3)

        with tf.name_scope("conv_R64"):
            conv1 = default_conv(self.stack, 64)
            r1 = tf.nn.leaky_relu(conv1, name="leaky_relu")

        with tf.name_scope("conv_BR128"):
            conv2 = default_conv(r1, 128)
            batch_norm1 = tf.layers.batch_normalization(conv2, training=True)
            r2 = tf.nn.leaky_relu(batch_norm1, name="leaky_relu")

        with tf.name_scope("conv_BR256"):
            conv3 = default_conv(r2, 256)
            batch_norm2 = tf.layers.batch_normalization(conv3, training=True)
            r3 = tf.nn.leaky_relu(batch_norm2, name="leaky_relu")

        with tf.name_scope("conv_BR512"):
            conv4 = default_conv(r3, 512)
            batch_norm3 = tf.layers.batch_normalization(conv4, training=True)
            r4 = tf.nn.leaky_relu(batch_norm3, name="leaky_relu")
        shape = r4.get_shape()
        num_features = shape[1:].num_elements()
        with tf.name_scope("fc"):
            reshape = tf.reshape(r4, [-1, num_features], name="reshape")
            W = tf.get_variable('weights', shape=[num_features, 1], initializer=tf.truncated_normal_initializer(stddev=0.2))
            b = tf.get_variable('biases', shape=[1], initializer=tf.constant_initializer(0.0))
            self.sigmoid = tf.nn.sigmoid(tf.matmul(reshape, W) + b, name="sigmoid")


