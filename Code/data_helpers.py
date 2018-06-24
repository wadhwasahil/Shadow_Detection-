import cv2
import os
import tensorflow as tf
import numpy as np
import random
import traceback


def resize(img, h=256, w=256, is_divide=True):
    if is_divide:
        return cv2.resize(img, (h, w)) / 255.
    return cv2.resize(img, (h, w))


avg_patches = []
w = 0.7
w_array = np.full((256, 256, 1), 0.7)

# tf.InteractiveSession()


def generate_patches(image, file_name, data_path="../Data/SBU-shadow/Train/ShadowImages", is_shadow=False):
    print(image.shape, file_name, "*"*10)
    if is_shadow:
     image_resized = np.array([np.expand_dims(image, -1)])
    else:
        image_resized = np.array([image])  # 1 * h * w * 3
    print(image_resized.shape, "*"*10)
    cnt = 1
    try:
        patches_256 = tf.extract_image_patches(image_resized, ksizes=[1, 256, 256, 1],
                                               strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
        if not is_shadow:
            patches_256_updated = tf.reshape(patches_256, [-1, 256, 256, 3])
        else:
            patches_256_updated = tf.reshape(patches_256, [-1, 256, 256, 1])
        n = patches_256_updated.get_shape().as_list()[0]
        patches_256_eval = patches_256_updated.eval()
        print(n, "^"*20)
        for i in range(n):
            patch_path = os.path.join(data_path, file_name.split(".png")[0] + "_" + str(cnt) + ".png")
            cnt += 1
            print(patch_path, patches_256_eval[i,].shape, "*"*20)
            cv2.imwrite(patch_path, patches_256_eval[i,])
        print("Done1")
    except:
        traceback.print_exc()
    try:
        h_new, w_new = int(image.shape[0] * 0.75), int(image.shape[1] * 0.75)
        patches_three_quarter = tf.extract_image_patches(image_resized, ksizes=[1, h_new, w_new, 1],
                                                         strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
        if not is_shadow:
            patches_three_quarter_updated = tf.reshape(patches_three_quarter, [-1, h_new, w_new, 3])
        else:
            patches_three_quarter_updated = tf.reshape(patches_three_quarter, [-1, h_new, w_new, 1])
        n = patches_three_quarter_updated.get_shape().as_list()[0]
        patches_three_quarter_eval = patches_three_quarter_updated.eval()
        for i in range(n):
            patch_path = os.path.join(data_path, file_name.split(".png")[0] + "_" + str(cnt) + ".png")
            print(patch_path, patches_three_quarter_eval[i,].shape, "*" * 20)
            cnt += 1
            cv2.imwrite(patch_path, resize(patches_three_quarter_eval[i,], is_divide=False))

        print("Done2")
    except:
        traceback.print_exc()



def create_patches(image):
    try:
        image_reshaped = np.array([image])
        patches_image_256 = tf.extract_image_patches(image_reshaped, ksizes=[1, 256, 256, 1],
                                                     strides=[1, 20, 20, 1], rates=[1, 1, 1, 1], padding="VALID")
        patches_image_256_updated = tf.reshape(patches_image_256, [-1, 256, 256, 3])
        n = patches_image_256_updated.get_shape().as_list()[0]
        patches_image_256_eval = patches_image_256_updated.eval()
        s2 = []
        for i in range(n):
            if i % 10 == 0:
                s2.append(patches_image_256_eval[i,])

        h_new, w_new = int(image.shape[0] * 0.75), int(image.shape[1] * 0.75)
        patches_image_three_quarter = tf.extract_image_patches(image_reshaped, ksizes=[1, h_new, w_new, 1],
                                                               strides=[1, 20, 20, 1], rates=[1, 1, 1, 1],
                                                               padding="VALID")
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

'''Remove resize when creating patches.'''
def read_image(path, is_color=True):
    if not is_color:
        return resize(cv2.threshold(cv2.imread(path, cv2.IMREAD_GRAYSCALE), 128, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    return np.concatenate((resize(cv2.imread(path)), w_array), axis=2)

def check_image(img):
    img_sum = np.sum(img)
    if img_sum:
        return True
    return False

def read_data(data_path="../Data/SBU-shadow", epochs=1, batch_size=16, train=True):
    if train:
        path = os.path.join(data_path, "Train")

        files = os.listdir(os.path.join(path, "ShadowImages2"))
        l = len(files)
        n_batches = l // batch_size
        print("Number of batches - ", n_batches)
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
                        abs_image_path = path + "/ShadowImages2/" + f
                        abs_shadow_path = path + "/ShadowMasks/" + f.split(".jpg")[0] + ".png"
                        # abs_shadow_path = path + "/ShadowMasks/" + f
                        cnt += 1

    #                     '''Generating patches'''
    #                     img = read_image(abs_image_path, is_color=True)
    #                     shadow_img = read_image(abs_shadow_path, is_color=False)
    #                     generate_patches(img[:, :, :3], f, data_path=path + "/ShadowImages/")
                        # generate_patches(shadow_img, f, data_path=path + "/ShadowMasks/", is_shadow=True)
                        img = read_image(abs_image_path)
                        shadow = read_image(abs_shadow_path, False)
                        if check_image(shadow):
                            arr.append((img, shadow))
                    except Exception as err:
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


