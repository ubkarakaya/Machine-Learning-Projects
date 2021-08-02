# -*- coding: utf-8 -*-

from scipy.misc import imread, imresize
import os
from os import listdir
from os.path import isfile, join
import random
from scipy import io
import cv2


def read_data():
    img_array = []
    img_label = []
    folders = {"daisy": 0, "dandelion": 1, "rose": 2, "sunflower": 3, "tulip": 4}
    for flower in folders.keys():
        data_path = os.path.join("../flowers/", flower)
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        for file in onlyfiles:
            img_file = os.path.join(data_path + "/", file)
            img = imread(img_file)
            img = imresize(img, [32, 24])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.flatten()
            img_array.append(img)
            img_label.append(folders[flower])
    #    print(len(img_array))
    c = list(zip(img_array, img_label))

    random.shuffle(c)

    img_array, img_label = zip(*c)
    data = {'x': img_array[0:3600], 'y': img_label[0:3600]}
    io.savemat('train.mat', data)
    #data = {'x': img_array[3001:3600], 'y': img_label[3001:3600]}
    #io.savemat('validation.mat', data)
    data = {'x': img_array[3601:4323], 'y': img_label[3601:4323]}
    io.savemat('test.mat', data)


# data = io.loadmat('test.mat')
