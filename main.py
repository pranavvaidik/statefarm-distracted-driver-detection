from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from glob import glob

#defining a function to load the dataset
def load_dataset(path):
    data = load_files(path)
    driver_images = np.array(data['filenames'])
    driver_activities = np_utils.to_categorical(np.array(data['target']))
    return driver_images, driver_activities

#loading the datasets
#change the directory in the github as well
images, targets = load_dataset('../input/state-farm-distracted-driver-detection/train/')

#splitting data to train, test and validation datasets
images_train, images_rest, targets_train, targets_rest = train_test_split( images, targets, train_size=0.8, random_state=42)
images_val, images_test, targets_val, targets_test = train_test_split( images_rest, targets_rest, train_size=0.5, random_state=42)


#printing the dataset statistics
print('There are %d total number of driver images' % len(images))


print('There are %d number of train images' % len(images_train))
print('There are %d number of validation images' % len(images_val))
print('There are %d number of test images' % len(images_test))


import os
print(os.listdir("../input/state-farm-distracted-driver-detection/train"))






###########################################################






