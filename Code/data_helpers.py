import cv2
import os
import tensorflow as tf
import numpy as np


def resize(img, h=256, w=256):
    return cv2.resize(img, (h, w))


def batch_iter(doc, batch_size, num_epochs, shuffle=True):
    """
    Generates batch iterator for a dataset.
    """
    data = list()
    for iter in doc:
        data.append(iter)
    # print("len", len(data))
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def generate_patches(image):
    sess = tf.InteractiveSession()
    data = []
    data.append(resize(image))
    print(image.shape)
    res = cv2.resize(image, None, fx=0.75, fy=0.75)
    image_resized = np.array([image]) # 1 * h * w * 3
    patches_256 = tf.extract_image_patches(image_resized, ksizes=[1, 256, 256, 1],
                                           strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
    patches_256 = tf.reshape(patches_256, [-1, 256 * 256 * 3])
    print(patches_256)
    patch1 = patches_256[0, ]
    patch1 = tf.reshape(patch1, [256, 256, 3])
    cv2.imshow('patch1', patch1.eval())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_data(data_path="../Data/SBU-shadow", train=True):
    if train:
        path = os.path.join(data_path, "Train")
    else:
        path = os.path.join(data_path, "Test")

    for _, _, files in os.walk(os.path.join(path, "ShadowImages")):
        for i, file in enumerate(files):
            try:
                abs_path_image = path + "/ShadowImages/" + file
                abs_path_shadow = path + "/ShadowMasks/" + file
                abs_path_shadow = abs_path_shadow.split(".jpg")[0] + ".png"
                img_data =  generate_patches(cv2.imread(abs_path_image))
                shadow = resize(cv2.imread(abs_path_shadow, cv2.IMREAD_GRAYSCALE))
            except Exception as e:
                print(file, "couldn't open properly")


im = cv2.imread("../Data/SBU-shadow/Train/ShadowImages/lssd3.jpg")
generate_patches(im)
