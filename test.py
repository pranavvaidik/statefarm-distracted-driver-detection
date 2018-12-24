from keras.preprocessing.image import ImageDataGenerator
from custom_functions import generate_bottleneck_features
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing import image                  
from tqdm import tqdm
import numpy as np


bottleneck_features_train, bottleneck_features_validation, bottleneck_features_test = generate_bottleneck_features()



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
