import tensorflow as tf

checkpoint_dir = "../Models/checkpoints"
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        X = graph.get_operation_by_name("X").outputs[0]
        ground_truth_shadow_masks = graph.get_operation_by_name("y").outputs[0]


        g_tanh = graph.get_operation_by_name("generator/deconv_1/tanh").outputs[0]

