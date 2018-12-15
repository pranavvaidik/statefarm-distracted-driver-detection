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

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs= epochs, callbacks=[checkpointer], use_multiprocessing=True, workers=6)


















