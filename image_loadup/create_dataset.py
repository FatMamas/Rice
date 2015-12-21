import os
from os import path
from random import shuffle
import numpy as np
from PIL import Image
import pickle

__author__ = 'Gyfis'

DATA_DIR = "../data/images"

#####
# 0: #beachporn
# 1: #burger
# 2: #catstagram
# 3: #moonporn
# 4: #nailsporn
# 5: #nikeporn
# 6: #puglove
# 7: #redphonebooth
# 8: #sushirolls
# 9: #yellowtulips


def image_to_array(filename):
    img = Image.open(filename)
    img.load()
    return np.asarray(img, dtype="int8").transpose(2, 0, 1)


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

    test_images = np.zeros((len(test_pairs), 3, 32, 32))
    test_types = [0]*len(test_pairs)
    i = 0
    for (image_array, image_type) in test_pairs:
        test_images[i] = image_array
        test_types[i] = image_type
        i += 1

    train_images = np.zeros((len(train_pairs), 3, 32, 32))
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


def save_img(img, f_name):
    from PIL import Image
    swapped_img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 0)
    Image.fromarray(swapped_img, 'RGB').save(f_name)


if __name__ == "__main__":
    save_img(image_to_array("0a6c16aeb3457dce4e4594005349df8423a10b46.jpg"), 'test.jpg')
    save_img(image_to_array("0a77ac9bfffe8c93d73c7311201fb6c7102581cb.jpg"), 'test3.jpg')
    save_img(image_to_array("Screen Shot 2015-12-21 at 08.46.30.png"), 'test2.png')

