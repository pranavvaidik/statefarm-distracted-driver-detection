from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from glob import glob
from data_gen import DataGenerator

#defining a function to load the dataset
def load_dataset(path):
	data = load_files(path)
	driver_images = np.array(data['filenames'])
	driver_activities = np_utils.to_categorical(np.array(data['target']))
	return driver_images, driver_activities

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


import os
print(os.listdir("imgs/train"))

################################################################

# Parameters for the Data Generator
params = {'batch_size' : 64,
          'shuffle': True}

# creating generators
training_generator = DataGenerator(images_train, targets_train, **params)
validation_generator = DataGenerator(images_val, targets_val, **params)
testing_generator = DataGenerator(images_test, targets_test, **params)

################################################################



#directly training with entire data

from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
	# loads RGB image as PIL.Image.Image type
	img = image.load_img(img_path, target_size=(224, 224))
	# convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
	x = image.img_to_array(img)
	# convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
	return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
	list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
	return np.vstack(list_of_tensors)


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(images_train).astype('float32')/255
valid_tensors = paths_to_tensor(images_val).astype('float32')/255
test_tensors = paths_to_tensor(images_test).astype('float32')/255


#################################################################










# Constructing Vanilla model

from  keras.layers  import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.

model.add(Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = None, input_shape = (256,256,3)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = None))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#model.add(Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#model.add(Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu'))
model.add(GlobalAveragePooling2D())

#model.add(Dropout(0.5))

model.add(Dense(10,activation = 'softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])






# Training the vanilla model

from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 25

### .

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.vanilla.hdf5', 
                               verbose=1, save_best_only=True)

#model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs= epochs, callbacks=[checkpointer], use_multiprocessing=True, workers=6)

#direct_train
model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

#load model with best validation loss
model.load_weights('saved_models/weights.best.from_scratch.vanilla.hdf5')


dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


"""
# get index of predicted dog breed for each image in test set
test_accuracy = model.evaluate_generator(generator=testing_generator,use_multiprocessing=True, workers=6)#[np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
#test_accuracy = 100*np.sum(np.array(action_predictions)==np.argmax(test_targets, axis=1))/len(action_predictions)
print('Test accuracy: %.4f%%' % test_accuracy[0])
"""















