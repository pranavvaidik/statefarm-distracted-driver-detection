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


################################################################


#directly training with entire data


from keras.preprocessing import image                  
from tqdm import tqdm

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


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 


images_train, images_test, images_val, targets_train, targets_test, targets_val = build_training_sets()
print(" Loaded the train, test and validation sets successfully!! ")

# pre-process the data for Keras
train_tensors = paths_to_tensor(images_train).astype('float32')/255
valid_tensors = paths_to_tensor(images_val).astype('float32')/255
test_tensors = paths_to_tensor(images_test).astype('float32')/255





from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

# define VGG16 model
vgg_model = VGG16(weights='imagenet', include_top = False)

vgg_model.summary()

vgg_layers = [l for l in vgg_model.layers]

trimmed_model = Sequential()

for i in range(len(vgg_layers)-12):
	vgg_layers[i].trainable = False
	trimmed_model.add(vgg_layers[i])

print("Trimmed")
trimmed_model.summary()

trimmed_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# get bottleneck features
try:
	bottleneck_features_test = np.load('bottleneck_features_test.npy')
	print("Bottleneck features for test set loaded")
except:
	print("Bottleneck features for test set not found")
	bottleneck_features_test = trimmed_model.predict(test_tensors)
	np.save('bottleneck_features_test.npy', bottleneck_features_test)

try:
	bottleneck_features_train = np.load('bottleneck_features_train.npy')
	print("Bottleneck features for train set loaded")

except:
	print("Bottleneck features for train set not found")
	bottleneck_features_train = trimmed_model.predict(train_tensors)
	np.save('bottleneck_features_train.npy', bottleneck_features_train)

try:
	bottleneck_features_val = np.load('bottleneck_features_validation.npy')
	print("Bottleneck features for validation set loaded")
except:
	print("Bottleneck features for validation set not found")
	bottleneck_features_val = timmed_model.predict(valid_tensors)
	np.save('bottleneck_features_validation.npy', bottleneck_features_val)




print("Bottleneck features obtained")

#build top model

top_model = Sequential()
top_model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))#vgg_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='sigmoid'))


top_model.summary()

top_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the top layers


print("Now, training top layers : ")

from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10

###

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.vgg16.hdf5', 
                               verbose=1, save_best_only=True)

#model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs= epochs, callbacks=[checkpointer], use_multiprocessing=True, workers=6)

#direct_train
top_model.fit(bottleneck_features_train, targets_train, 
          validation_data=(bottleneck_features_val, targets_val),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

#load model with best validation loss
top_model.load_weights('weights.best.from_scratch.vgg16.hdf5')

val_pred = [np.argmax(top_model.predict(np.expand_dims(tensor, axis=0))) for tensor in bottleneck_features_val]



predictions = [np.argmax(top_model.predict(np.expand_dims(tensor, axis=0))) for tensor in bottleneck_features_test]


val_accuracy = 100*np.sum(np.array(val_pred)==np.argmax(targets_val, axis=1))/len(val_pred)
print('Validation accuracy: %.4f%%' % val_accuracy)


# report test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(targets_test, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)











