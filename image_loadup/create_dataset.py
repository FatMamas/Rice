import os
from os import path
from random import shuffle
import numpy as np
from PIL import Image
import pickle

__author__ = 'Gyfis'

DATA_DIR = "../data/images"


def image_to_array(filename):
    img = Image.open(filename)
    img.load()
    arr = np.asarray(img, dtype="int8").transpose(2, 0, 1)
    print(arr.shape)
    return np.reshape(arr, arr.shape[0] * arr.shape[1] * arr.shape[2])


def load_and_save_images():
    test_size = 800

    train_pairs = []
    test_pairs = []
    for directory in [x for x in os.listdir(DATA_DIR) if path.isdir(DATA_DIR + os.sep + x)]:
        i = 0
        for image in [x for x in os.listdir(DATA_DIR + os.sep + directory) if path.isfile(DATA_DIR + os.sep + directory + os.sep + image)]:
            if i < test_size:
                test_pairs.append((image_to_array(image), int(directory)))
                i += 1
            else:
                train_pairs.append((image_to_array(image), int(directory)))

    shuffle(test_pairs)
    shuffle(train_pairs)

    test_images = np.zeros((len(test_pairs), 3072))
    test_types = [0]*len(test_pairs)
    i = 0
    for (image_array, image_type) in test_pairs:
        test_images[i] = image_array
        test_types[i] = image_type
        i += 1

    train_images = np.zeros((len(train_pairs), 3072))
    train_types = [0]*len(train_pairs)
    i = 0
    for (image_array, image_type) in train_pairs:
        train_images[i] = image_array
        train_types[i] = image_type
        i += 1

    test_dataset = {'data': test_images, 'labels': test_types}
    pickle.dump(test_dataset, open('test_batch'))

    train_dataset = {'data': train_images, 'labels': train_types}
    pickle.dump(train_dataset, open('train_batch'))


if __name__ == "__main__":
    print(image_to_array("0a6c16aeb3457dce4e4594005349df8423a10b46.jpg").shape)
    print(image_to_array("Screen Shot 2015-12-21 at 08.46.30.png").shape)

