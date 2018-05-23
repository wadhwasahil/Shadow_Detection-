import tensorflow as tf
import numpy as np
from data_helpers import read_data
from sklearn.metrics import confusion_matrix
import cv2

TP = 0.
FP = 0.
TN = 0.
FN = 0.


def resize(img, h=256, w=256):
    return cv2.resize(img, (h, w))


checkpoint_dir = "../Models/1527095768/checkpoints/"
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
batch_size = 16


def update_confusion_matrix(orig, pred):
    print(pred.shape)
    for i in range(batch_size):
        orig_expanded = orig[i].flatten()
        pred_expanded = pred[i].flatten()
        print(orig_expanded.shape, pred_expanded.shape)
        tn, fp, fn, tp = confusion_matrix(orig_expanded, pred_expanded).ravel()
        TP = TP + tp
        FP = FP + fp
        TN = TN + tn
        FN = FN + fn


def BER():
    return 1. - 0.5 * ((TP / (TP + FN)) + (TN / (TN + FP)))


graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        X = graph.get_operation_by_name("X").outputs[0]
        ground_truth_shadow_masks = graph.get_operation_by_name("y").outputs[0]

        g_tanh = graph.get_operation_by_name("generator/deconv_1/tanh").outputs[0]

        for i, batch in enumerate(read_data(train=False)):
            s1, s2, s3, shadow = np.array([batch[0]]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3])
            print(s1.shape, s2.shape, s3.shape, shadow)
            orig_w, orig_h = shadow.shape
            denominator = 25 + 5 * s2.shape[0] + s3.shape[0]
            s1_shadow_map = np.array(sess.run(g_tanh, feed_dict={X: s1}))
            s2_shadow_map = np.array(sess.run(g_tanh, feed_dict={X: s2}))
            s3_shadow_map = np.array(sess.run(g_tanh, feed_dict={X: s3}))
            s1_shadow_map_resized = 25. * np.array(resize(s1_shadow_map, h=orig_h, w=orig_w))
            s2_shadow_map_resized = 5. * np.array(
                [resize(s2_shadow_map[k], h=orig_h, w=orig_w) for k in range(s2_shadow_map.shape[0])])
            s3_shadow_map_resized = np.array([resize(s3_shadow_map[k], h=orig_h, w=orig_w) for k in
                                              range(s3_shadow_map.shape[0])])
            weighted_matrx = s1_shadow_map_resized
            for k in range(s2_shadow_map_resized.shape[0]):
                weighted_matrx = weighted_matrx + s2_shadow_map_resized[k]
            for k in range(s3_shadow_map_resized.shape[0]):
                weighted_matrx = weighted_matrx + s3_shadow_map_resized[k]
            shadow_predicted = weighted_matrx / denominator
            # update_confusion_matrix(y, np.array(predicted_shadow_map))

            cv2.imshow("predicted shadow", shadow_predicted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

        # print(BER())
