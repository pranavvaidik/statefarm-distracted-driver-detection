from keras.preprocessing.image import ImageDataGenerator
from custom_functions import paths_to_tensor,  build_training_sets
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing import image                  
from tqdm import tqdm


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
bottleneck_features_train = trimmed_model.predict_generator(train_gen, steps = 100)
bottleneck_features_validation = trimmed_model.predict_generator(validation_gen, steps = 12)
bottleneck_features_test = trimmed_model.predict_generator(test_gen, steps = 12)




#np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
#np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
#np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)





"""
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

"""
