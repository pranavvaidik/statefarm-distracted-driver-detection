from keras.preprocessing.image import ImageDataGenerator
from custom_functions import generate_bottleneck_features, build_training_sets, paths_to_tensor
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing import image                  
from tqdm import tqdm
import numpy as np







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

#vgg_model.summary()

vgg_layers = [l for l in vgg_model.layers]

model = Sequential()

for layer in vgg_layers[:11]:
	layer.trainable = False
	model.add(layer)

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


"""

try:
	print("Attempting to load the bottleneck features")
	bottleneck_features_train = np.load('bottleneck_features_train.npy')
	bottleneck_features_validation = np.load('bottleneck_features_validation.npy')
	bottleneck_features_test = np.load('bottleneck_features_test.npy')
	print("Bottleeck features loaded successfully!!")
except:
	print("Bottleneck features were not found!")
	bottleneck_features_train, bottleneck_features_validation, bottleneck_features_test = generate_bottleneck_features()
"""


"""
#build top model

top_model = Sequential()
top_model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))#vgg_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='sigmoid'))


top_model.summary()

top_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
"""

# train the top layers


print("Now, training top layers : ")

from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10

###

checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.vgg16.hdf5', 
                               verbose=1, save_best_only=True)

model.fit_generator(generator=train_gen, steps_per_epoch=300, validation_data=validation_gen, validation_steps = 40, epochs= epochs, callbacks=[checkpointer], use_multiprocessing=True, workers=6, verbose=1)

#direct_train
#model.fit(bottleneck_features_train, targets_train, validation_data=(bottleneck_features_val, targets_val), epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

#load model with best validation loss
model.load_weights('weights.best.from_scratch.vgg16.hdf5')


valid_acc = model.evaluate_generator(generator = validation_gen, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
test_accuracy = model.evaluate_generator(generator = test_gen, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)




#val_pred = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in bottleneck_features_val]



#predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in bottleneck_features_test]


#val_accuracy = 100*np.sum(np.array(val_pred)==np.argmax(targets_val, axis=1))/len(val_pred)
print('Validation accuracy: %.4f%%' % valid_acc)
print('Test accuracy: %.4f%%' % test_accuracy)


