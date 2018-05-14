import cv2
import os
import tensorflow as tf
import numpy as np
import random


def resize(img, h=256, w=256):
    return cv2.resize(img, (h, w))


def generate_patches(image):
    sess = tf.InteractiveSession()
    data = []
    data.append(resize(image))
    res = cv2.resize(image, None, fx=0.75, fy=0.75)
    image_resized = np.array([image])  # 1 * h * w * 3
    print(image_resized.shape)
    patches_256 = tf.extract_image_patches(image_resized, ksizes=[1, 256, 256, 1],
                                           strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
    patches_256 = tf.reshape(patches_256, [-1, 256 * 256 * 3])
    print(patches_256)
    patch1 = patches_256[0,]
    patch1 = tf.reshape(patch1, [256, 256, 3])
    cv2.imshow('patch1', patch1.eval())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_image(path, is_color=True):
    if not is_color:
        return resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    return resize(cv2.imread(path))


def read_data(data_path="../Data/SBU-shadow", epochs=1, batch_size=16, train=True):
    if train:
        path = os.path.join(data_path, "Train")
    else:
        path = os.path.join(data_path, "Test")

    files = os.listdir(os.path.join(path, "ShadowImages"))
    l = len(files)
    n_batches = l // batch_size
    if l % batch_size != 0:
        n_batches = n_batches + 1
    if train:
        for e in range(epochs):
            shuffle_files = random.sample(files, l)
            for batch_num in range(n_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, l)
                arr = []
                for f in shuffle_files[start_index:end_index]:
                    try:
                        # todo - add patches per image
                        abs_image_path = path + "/ShadowImages/" + f
                        abs_shadow_path = path + "/ShadowMasks/" + f.split(".jpg")[0] + ".png"
                        arr.append((read_image(abs_image_path) / 255., read_image(abs_shadow_path, False) / 255.))
                    except Exception as e:
                        pass
                yield arr


#
# im = cv2.imread("../Data/SBU-shadow/Train/ShadowImages/lssd3.jpg")
# generate_patches(im)

for i in read_data():
    print(i)
