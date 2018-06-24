import tensorflow as tf


def default_conv(input, num_filters):
    input_channels = input.get_shape().as_list()[-1]
    b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b" + str(num_filters))
    W = tf.Variable(tf.truncated_normal([5, 5, input_channels, num_filters], stddev=0.2), name="weights")
    conv = tf.nn.bias_add(tf.nn.conv2d(input, filter=W, strides=[1, 2, 2, 1], padding="SAME"), b, name="conv")
    return conv


# def default_deconv(input, num_filters):
#     b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
#     return tf.nn.bias_add(tf.layers.conv2d_transpose(input, num_filters, kernel_size=(5, 5),
#                                                      kernel_initializer=tf.random_normal_initializer(stddev=0.2),
#                                                      strides=(2, 2), padding="same"), b, name="deconv")


# def default_conv(input, num_filters):
#     b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
#     conv = tf.nn.bias_add(tf.layers.conv2d(input, filters=num_filters, kernel_size=(5, 5),
#                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
#                                            strides=(2, 2), padding="same"), b, name="conv")
#     return conv


def default_deconv(input, num_filters):
    b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
    return tf.nn.bias_add(tf.layers.conv2d_transpose(input, num_filters, kernel_size=(5, 5),
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                                                     strides=(2, 2), padding="same"), b, name="deconv")
class Generator(object):
    def __init__(self, X, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("conv64"):
                conv1 = default_conv(X, 64)
            with tf.variable_scope("conv_RB128"):
                r1 = tf.nn.leaky_relu(conv1, name="leaky_relu")
                conv2 = default_conv(r1, 128)
                batch_norm1 = tf.layers.batch_normalization(conv2, training=True)

            with tf.variable_scope("conv_RB256"):
                r3 = tf.nn.leaky_relu(batch_norm1, name="leaky_relu")
                conv3 = default_conv(r3, 256)
                batch_norm2 = tf.layers.batch_normalization(conv3, training=True)

            with tf.variable_scope("conv_RB512_1"):
                r4 = tf.nn.leaky_relu(batch_norm2, name="leaky_relu")
                conv4 = default_conv(r4, 512)
                batch_norm3 = tf.layers.batch_normalization(conv4, training=True)

            with tf.variable_scope("conv_RB512_2"):
                r5 = tf.nn.leaky_relu(batch_norm3, name="leaky_relu")
                conv5 = default_conv(r5, 512)
                batch_norm4 = tf.layers.batch_normalization(conv5, training=True)

            with tf.variable_scope("conv_RB512_3"):
                r6 = tf.nn.leaky_relu(batch_norm4, name="leaky_relu")
                conv6 = default_conv(r6, 512)
                batch_norm5 = tf.layers.batch_normalization(conv6, training=True)

            with tf.variable_scope("conv_RB512_4"):
                r7 = tf.nn.leaky_relu(batch_norm5, name="leaky_relu")
                conv7 = default_conv(r7, 512)
                batch_norm6 = tf.layers.batch_normalization(conv7, training=True)

            with tf.variable_scope("conv_RB512_5"):
                r8 = tf.nn.leaky_relu(batch_norm6, name="leaky_relu")
                conv8 = default_conv(r8, 512)
                batch_norm7 = tf.layers.batch_normalization(conv8, training=True)

            with tf.variable_scope("deconv_DR512_1"):
                r9 = tf.nn.relu(batch_norm7, name="relu")
                deconv1 = default_deconv(r9, 512)
                batch_norm8 = tf.layers.batch_normalization(deconv1, training=True)
                dropout1 = tf.nn.dropout(batch_norm8, keep_prob=0.5, name="dropout")
                concat1 = tf.concat([dropout1, batch_norm6], axis=-1)

            with tf.variable_scope("deconv_DR512_2"):
                r10 = tf.nn.relu(concat1, name="relu")
                deconv2 = default_deconv(r10, 512)
                batch_norm9 = tf.layers.batch_normalization(deconv2, training=True)
                dropout2 = tf.nn.dropout(batch_norm9, keep_prob=0.5, name="dropout")
                concat2 = tf.concat([dropout2, batch_norm5], axis=-1)

            with tf.variable_scope("deconv_DR512_3"):
                r11 = tf.nn.relu(concat2, name="relu")
                deconv3 = default_deconv(r11, 512)
                batch_norm10 = tf.layers.batch_normalization(deconv3, training=True)
                dropout3 = tf.nn.dropout(batch_norm10, keep_prob=0.5, name="dropout")
                concat3 = tf.concat([dropout3, batch_norm4], axis=-1)

            with tf.variable_scope("deconv_R512"):
                r12 = tf.nn.relu(concat3, name="relu")
                deconv4 = default_deconv(r12, 512)
                batch_norm11 = tf.layers.batch_normalization(deconv4, training=True)
                concat3 = tf.concat([batch_norm3, batch_norm11], axis=-1)

            with tf.variable_scope("deconv_R256"):
                r13 = tf.nn.relu(concat3, name="relu")
                deconv5 = default_deconv(r13, 256)
                batch_norm12 = tf.layers.batch_normalization(deconv5, training=True)
                concat4 = tf.concat([batch_norm2, batch_norm12], axis=-1)

            with tf.variable_scope("deconv_R128"):
                r14 = tf.nn.relu(concat4, name="relu")
                deconv6 = default_deconv(r14, 128)
                batch_norm13 = tf.layers.batch_normalization(deconv6, training=True)
                concat5 = tf.concat([batch_norm1, batch_norm13], axis=-1)

            with tf.variable_scope("deconv_R64"):
                r14 = tf.nn.relu(concat5, name="relu")
                deconv7 = default_deconv(r14, 64)
                batch_norm14 = tf.layers.batch_normalization(deconv7, training=True)
                concat4 = tf.concat([conv1, batch_norm14], axis=-1)

            # todo - tanh giving nan values
            with tf.variable_scope("deconv_1"):
                r15 = tf.nn.relu(concat4, name="relu")
                deconv8 = default_deconv(r15, 1)
                self.tanh = tf.nn.sigmoid(tf.squeeze(deconv8, name="tanh"))


