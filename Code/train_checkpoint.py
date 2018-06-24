import tensorflow as tf
from data_helpers import read_data
import numpy as np


batch_size = 16
gamma = 0.7
eps = 1e-8
lambda_G = 1.
learning_rate = 0.001

checkpoint_dir = "../Models/1529516502/checkpoints/"
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)


graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        try:
            print("Loading model from {}".format(checkpoint_file))
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print("Model successfully loaded...")
        except Exception as err:
            print("Error loading {}".format(checkpoint_file))
        X = graph.get_operation_by_name("X").outputs[0]

        ground_truth_shadow_masks = graph.get_operation_by_name("y").outputs[0]
        g_tanh = graph.get_operation_by_name("generator/deconv_1/tanh").outputs[0]
        d_sigmoid = graph.get_operation_by_name("discriminator/fc/sigmoid").outputs[0]
        global_step = graph.get_operation_by_name("global_step").outputs[0]
        d_optimizer = graph.get_operation_by_name("train/d_optimizer").outputs[0]
        g_optimizer = graph.get_operation_by_name("train/g_optimizer").outputs[0]

        for batch, e, num in read_data(epochs=1000):
            x, y = zip(*batch)
            x = np.array(x)
            y = np.array(y)
            step, summary, d_loss_value = sess.run([d_optimizer, merged_summary, d_loss],
                                                   feed_dict={X: x, ground_truth_shadow_masks: y})