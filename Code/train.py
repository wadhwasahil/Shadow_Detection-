import tensorflow as tf
from data_helpers import read_data
from generator import Generator
from discriminator import Discriminator
import os
import time
import numpy as np

batch_size = 8
gamma = 0.7
eps = 1e-8
lambda_G = 10.
learning_rate = 0.0001

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        X = tf.placeholder(tf.float32, [None, 256, 256, 4], name="X")
        X_slice = X[:, :, :, :3]
        input_image = tf.summary.image("input_image", X_slice, max_outputs=3)
        ground_truth_shadow_masks = tf.placeholder(tf.float32, [None, 256, 256], name="y")
        shadow_image = tf.summary.image("shadow_image", tf.expand_dims(ground_truth_shadow_masks, -1), max_outputs=3)
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Generator
        gx = Generator(X).tanh  # shadow mask of size 256 * 256 by generator
        generator_image = tf.summary.image("shadow_by_generator", tf.expand_dims(gx, -1), max_outputs=1)

        # Discriminator
        dx_real = Discriminator(X_slice, ground_truth_shadow_masks).sigmoid
        dx_fake = Discriminator(X_slice, gx, reuse=True).sigmoid

        ground_truth_shadow_masks_flatten = tf.layers.flatten(ground_truth_shadow_masks)
        t1 = tf.scalar_mul(-gamma, tf.matmul(ground_truth_shadow_masks_flatten, tf.log(tf.reshape(gx, [-1, 256 * 256]) + eps), transpose_a=True))
        t2 = tf.scalar_mul(1 - gamma,
                           tf.matmul((1. - ground_truth_shadow_masks_flatten), tf.log(1. - tf.reshape(gx, [-1, 256 * 256]) + eps), transpose_a=True))

        L_data = tf.reduce_mean(t1 - t2)

        d_real_hist = tf.summary.histogram("d_real_hist", dx_real)
        d_fake_hist = tf.summary.histogram("d_fake_hist", dx_fake)

        # L_cGan = tf.reduce_mean(-tf.log(dx_real + eps) - tf.log(1. - dx_fake + eps))

        L_mse = tf.reduce_mean(tf.square(tf.reshape(gx, [-1, 256 * 256]) - ground_truth_shadow_masks_flatten),
                               name="mse")
        with tf.variable_scope("D_loss"):
            d_loss = -tf.reduce_mean(tf.log(dx_real + eps) + tf.log(1. - dx_fake + eps), name="d_loss_value")
            d_loss_summary = tf.summary.scalar("d_loss_summary", d_loss)

        with tf.variable_scope("G_loss"):
            g_loss = tf.add(-tf.reduce_mean(tf.log(dx_fake + eps)), lambda_G * L_mse, name="g_loss_value")
            # g_loss = tf.add(-tf.reduce_mean(tf.log(dx_fake + eps)), , name="g_loss_value")
            g_loss_summary = tf.summary.scalar("g_loss_summary", g_loss)

        tvar = tf.trainable_variables()
        dvar = [var for var in tvar if 'discriminator' in var.name]
        gvar = [var for var in tvar if 'generator' in var.name]

        with tf.variable_scope('train'):
            d_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=dvar,
                                                                                        name="d_optimizer")
            g_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=gvar,
                                                                                        name="g_optimizer")
        init = tf.global_variables_initializer()
        sess.run(init)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join("../", "Models", timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        g_summary = tf.summary.merge([input_image, generator_image, shadow_image, g_loss_summary, d_fake_hist])
        d_summary = tf.summary.merge([d_loss_summary, d_real_hist])
        # merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(out_dir + "/summaries")
        writer.add_graph(sess.graph)
        saver = tf.train.Saver(tf.all_variables())

        cnt = 0
        for batch, e, num in read_data(
                # data_path="/home/sahil/Desktop/Projects/Shadow_Detection_DL/Data/Videos/aton_campus/data",
                batch_size=batch_size, epochs=1000):
            x, y = zip(*batch)
            x = np.array(x)
            y = np.array(y)
            step, summary, d_loss_value, d1, d2 = sess.run([d_train_step, d_summary, d_loss, dx_real, dx_fake],
                                                   feed_dict={X: x, ground_truth_shadow_masks: y})
            writer.add_summary(summary, cnt)
            step, summary, g_loss_value, d1, d2 = sess.run([g_train_step, g_summary, g_loss, dx_real, dx_fake],
                                                   feed_dict={X: x, ground_truth_shadow_masks: y})
            writer.add_summary(summary, cnt)
            step, summary, g_loss_value, d1, d2 = sess.run([g_train_step, g_summary, g_loss, dx_real, dx_fake],
                                                   feed_dict={X: x, ground_truth_shadow_masks: y})
            writer.add_summary(summary, cnt)
            if cnt > 0 and cnt % 7 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))
            cnt += 1
            print("Epoch {:g}".format(e), "Batch Number {:g}".format(num), "D_Loss {:g}".format(d_loss_value),
                  "G_Loss {:g}".format(g_loss_value), d1, d2)
