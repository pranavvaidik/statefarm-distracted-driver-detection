from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from glob import glob
from data_gen import DataGenerator
from keras.preprocessing import image                  
from tqdm import tqdm

#defining a function to load the dataset
def load_dataset(path):
	data = load_files(path)
	driver_images = np.array(data['filenames'])
	driver_activities = np_utils.to_categorical(np.array(data['target']))
	return driver_images, driver_activities




def build_training_sets():

	#loading the datasets
	#change the directory in the github as well
	images, targets = load_dataset('imgs/train/')

	#splitting data to train, test and validation datasets
	images_train, images_rest, targets_train, targets_rest = train_test_split( images, targets, train_size=0.8, random_state=42)
	images_val, images_test, targets_val, targets_test = train_test_split( images_rest, targets_rest, train_size=0.5, random_state=42)


	#printing the dataset statistics
	print('There are %d total number of driver images' % len(images))


	print('There are %d number of train images' % len(images_train))
	print('There are %d number of validation images' % len(images_val))
	print('There are %d number of test images' % len(images_test))


	#import os
	#print(os.listdir("imgs/train"))

	return images_train, images_test, images_val, targets_train, targets_test, targets_val

def path_to_tensor(img_path):
	# loads RGB image as PIL.Image.Image type
	img = image.load_img(img_path, target_size=(256, 256))
	# convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
	x = image.img_to_array(img)
	# convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
	return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
	list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
	return np.vstack(list_of_tensors)
