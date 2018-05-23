import cv2
import os
import tensorflow as tf
import numpy as np
import random
import traceback

def resize(img, h=256, w=256):
    return cv2.resize(img, (h, w)) / 255.

avg_patches = []
w = 0.7
w_array = np.full((256, 256, 1), 0.7)

tf.InteractiveSession()

# def generate_patches(image, file_name, data_path="../Data/SBU-shadow/Train/ShadowImages"):
#     data = []
#     data.append(resize(image))
#     image_resized = np.array([image])  # 1 * h * w * 3
#     cnt = 1
#     try:
#         patches_256 = tf.extract_image_patches(image_resized, ksizes=[1, 256, 256, 1],
#                                                strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
#         patches_256_updated = tf.reshape(patches_256, [-1, 256, 256, 3])
#         n = patches_256_updated.get_shape().as_list()[0]
#         patches_256_eval = patches_256_updated.eval()
#         for i in range(n):
#             patch_path = os.path.join(data_path, file_name.split(".jpg")[0] + "_" + str(cnt) + ".jpg")
#             cnt += 1
#             cv2.imwrite(patch_path, patches_256_eval[i, ])
#         print("Done1")
#     except:
#         pass
#     try:
#         h_new, w_new = int(image.shape[0] * 0.75), int(image.shape[1] * 0.75)
#         patches_three_quarter = tf.extract_image_patches(image_resized, ksizes=[1, h_new, w_new, 1],
#                                                strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
#         patches_three_quarter_updated = tf.reshape(patches_three_quarter, [-1, h_new, w_new, 3])
#         n = patches_three_quarter_updated.get_shape().as_list()[0]
#         patches_three_quarter_eval = patches_three_quarter_updated.eval()
#         for i in range(n):
#             patch_path = os.path.join(data_path, file_name.split(".jpg")[0] + "_" + str(cnt) + ".jpg")
#             cnt += 1
#             cv2.imwrite(patch_path, resize(patches_three_quarter_eval[i, ]))
#         print("Done2")
#     except:
#         pass
    # cv2.imshow('patch1', patches_three_quarter_updated[0, ].eval())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def create_patches(image):
    try:
        image_reshaped = np.array([image])
        patches_image_256 = tf.extract_image_patches(image_reshaped, ksizes=[1, 256, 256, 1],
                                               strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
        patches_image_256_updated = tf.reshape(patches_image_256, [-1, 256, 256, 3])
        n = patches_image_256_updated.get_shape().as_list()[0]
        patches_image_256_eval = patches_image_256_updated .eval()
        s2 = []
        for i in range(n):
            if i % 10 == 0:
              s2.append(patches_image_256_eval[i, ])

        h_new, w_new = int(image.shape[0] * 0.75), int(image.shape[1] * 0.75)
        patches_image_three_quarter = tf.extract_image_patches(image_reshaped, ksizes=[1, h_new, w_new, 1],
                                                     strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
        patches_image_three_quarter_updated = tf.reshape(patches_image_three_quarter, [-1, h_new, w_new, 3])
        n = patches_image_three_quarter_updated.get_shape().as_list()[0]
        patches_image_three_quarter_eval = patches_image_three_quarter_updated.eval()
        s3 = []
        for i in range(n):
            if i % 10 == 0:
                s3.append(resize(patches_image_three_quarter_eval[i,]))
        return (s2, s3)
    except:
        pass

def read_image(path, is_color=True):
    if not is_color:
        return resize(cv2.threshold(cv2.imread(path, cv2.IMREAD_GRAYSCALE), 128, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    return np.concatenate((resize(cv2.imread(path)), w_array), axis=2)



def read_data(data_path="../Data/SBU-shadow", epochs=1, batch_size=16, train=True):
    if train:
        path = os.path.join(data_path, "Train")

        files = os.listdir(os.path.join(path, "ShadowImages"))
        l = len(files)
        n_batches = l // batch_size
        if l % batch_size != 0:
            n_batches = n_batches + 1
        for e in range(epochs):
            # shuffle_files = random.sample(files, l)
            shuffle_files = files
            cnt = 0
            for batch_num in range(n_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, l)
                arr = []
                for f in shuffle_files[start_index:end_index]:
                    try:
                        abs_image_path = path + "/ShadowImages/" + f
                        abs_shadow_path = path + "/ShadowMasks/" + f.split(".jpg")[0] + ".png"
                        cnt += 1
                        # img = read_image(abs_image_path)
                        # generate_patches(img, f)
                        arr.append((read_image(abs_image_path), read_image(abs_shadow_path, False)))
                    except Exception as e:
                        traceback.print_exc()
                yield arr, e, batch_num

    else:
        path = os.path.join(data_path, "Test")
        files = os.listdir(os.path.join(path, "ShadowImages"))
        for batch_num in range(len(files)):
            try:
                abs_image_path = path + "/ShadowImages/" + files[batch_num]
                abs_shadow_path = path + "/ShadowMasks/" + files[batch_num].split(".jpg")[0] + ".png"
                print("Opening ", abs_image_path, "." * 10)
                img = cv2.imread(abs_image_path)
                shadow = (cv2.threshold(cv2.imread(abs_shadow_path, cv2.IMREAD_GRAYSCALE), 128, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]) / 255.
                s1 = read_image(abs_image_path, is_color=True)
                s2, s3 = create_patches(img)
                s2 = [np.concatenate((k, w_array), axis=2) for k in s2]
                s3 = [np.concatenate((k, w_array), axis=2) for k in s3]
                yield np.array(s1), np.array(s2), np.array(s3), np.array(shadow)
            except Exception as e:
                traceback.print_exc()


