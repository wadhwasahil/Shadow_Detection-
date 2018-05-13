import tensorflow as tf
import data_helpers
from generator import Generator
from discriminator import Discriminator
# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#     session = tf.Session(config=session_conf)
#
#     # todo image augmentation
#     with session.as_default():
gen = Generator()
dis = Discriminator()
global_step = tf.Variable(0, name="global_step", trainable=False)
y = tf.placeholder(tf.float32, [None, 256, 256, 1])
optimizer = tf.train.AdamOptimizer(1e-3)

# Generator
w = 0.1
gx = gen.tanh
t1 = tf.scalar_mul(-w, tf.matmul(y, tf.log(gx), transpose_a=True))
t2 = tf.scalar_mul(w - 1, tf.matmul((1 - y), tf.log(1 - gx), transpose_a=True))

data_loss = tf.reduce_mean(t1 + t2)




