from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from glob import glob
from data_gen import DataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing import image                  
from tqdm import tqdm
import keras

#defining a function to load the dataset
def load_dataset(path):
	data = load_files(path)
	driver_images = np.array(data['filenames'])
	driver_activities = np_utils.to_categorical(np.array(data['target']))
	return driver_images, driver_activities



class DataGenerator(keras.utils.Sequence):

	#def __init__(self, list_file_paths, labels, batch_size=32, dim=(32,32,32), n_channels=1,
	#             n_classes=10, shuffle=True):
	def __init__(self, list_file_paths, labels, batch_size=32, shuffle=True):

		#Initialization
		self.batch_size = batch_size
		self.labels = labels
		self.list_file_paths = list_file_paths
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		#Denotes the number of batches per epoch
		return int(np.floor(len(self.list_file_paths) / self.batch_size))

	def __getitem__(self, index):
		#Generate one batch of data
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_file_paths[k] for k in indexes]

		# Generate data
		X = self.__data_generation(list_IDs_temp)
		y = self.labels[indexes]

		return X, y

	def on_epoch_end(self):
		#Updates indexes after each epoch
		self.indexes = np.arange(len(self.list_file_paths))
		if self.shuffle == True:
	    		np.random.shuffle(self.indexes)

	def path_to_tensor(self, path):
		img = image.load_img(path, target_size=(256, 256))
		x = image.img_to_array(img)
		return np.expand_dims(x, axis=0)

	#define a function that reads a path to an image and returns a tensor suitable for keras
	#def paths_to_tensor(paths_list):
	#    img_list = [path_to_tensor(img_path) for img_path in tqdm(paths_list)]
	#    return np.vstack(img_list)

	def __data_generation(self, list_file_paths_temp):

		#Generates data containing batch_size samples
		# X : (n_samples, *dim, n_channels)
		# Initialization
		img_list = [self.path_to_tensor(img_path) for img_path in list_file_paths_temp]
		return np.vstack(img_list).astype('float32')/255





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
	
	
def generate_bottleneck_features():
	images_train, images_test, images_val, targets_train, targets_test, targets_val = build_training_sets()
	print(" Loaded the train, test and validation sets successfully!! ")

	# pre-process the data for Keras
	train_tensors = paths_to_tensor(images_train).astype('float32')/255
	valid_tensors = paths_to_tensor(images_val).astype('float32')/255
	test_tensors = paths_to_tensor(images_test).astype('float32')/255


	datagen = ImageDataGenerator(
	    rotation_range=20,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=False)
	    
	train_gen = datagen.flow(train_tensors, targets_train, batch_size = 128)
	test_gen = datagen.flow(test_tensors, targets_test, batch_size = 128)
	validation_gen = datagen.flow(valid_tensors, targets_val, batch_size = 128)

	# define VGG16 model
	vgg_model = VGG16(weights='imagenet', include_top = False, input_shape = (256,256,3))

	vgg_model.summary()

	vgg_layers = [l for l in vgg_model.layers]

	trimmed_model = Sequential()

	for layer in vgg_layers[:11]:
		layer.trainable = False
		trimmed_model.add(layer)

	print("Trimmed")
	trimmed_model.summary()
	trimmed_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	## generating bottleneck features as generator objetcs, steps are based on the  number of tensors in each set and also the data augmentation
	#bottleneck_features_train = trimmed_model.predict_generator(train_gen, steps = 150)
	#bottleneck_features_validation = trimmed_model.predict_generator(validation_gen, steps = 20)
	#bottleneck_features_test = trimmed_model.predict_generator(test_gen, steps = 20)




	#np.save('bottleneck_features_train.npy', bottleneck_features_train)
	#np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
	#np.save('bottleneck_features_test.npy', bottleneck_features_test)
	
	return vgg_model# bottleneck_features_train, bottleneck_features_validation, bottleneck_features_test

