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
	vgg_model = VGG16(weights='imagenet', include_top = False)

	vgg_model.summary()

	vgg_layers = [l for l in vgg_model.layers]

	trimmed_model = Sequential()

	for layer in vgg_layers[:11]:
		layer.trainable = False
		trimmed_model.add(layer)

	print("Trimmed")
	trimmed_model.summary()
	trimmed_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	# generating bottleneck features as generator objetcs, steps are based on the  number of tensors in each set and also the data augmentation
	bottleneck_features_train = trimmed_model.predict_generator(train_gen, steps = 150)
	bottleneck_features_validation = trimmed_model.predict_generator(validation_gen, steps = 20)
	bottleneck_features_test = trimmed_model.predict_generator(test_gen, steps = 20)




	np.save('bottleneck_features_train.npy', bottleneck_features_train)
	np.save('bottleneck_features_validation.npy' bottleneck_features_validation)
	np.save('bottleneck_features_test.npy', bottleneck_features_test)
	
	return bottleneck_features_train, bottleneck_features_validation, bottleneck_features_test

