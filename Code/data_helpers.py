import cv2
import os

def resize(img, h=256, w=256):
    return cv2.resize(img, (h,w))


def read_data(data_path="../Data/SBU-shadow", train=True, batch_size=16):
    if train:
        path = os.path.join(data_path, "Train")
    else:
        path = os.path.join(data_path, "Test")

    for _, _, files in os.walk(os.path.join(path, "ShadowImages")):
        for file in files:
            print(file)

