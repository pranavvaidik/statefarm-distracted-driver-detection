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


bottleneck_features_train = trimmed_model.predict_generator(train_gen)
bottleneck_features_validation = trimmed_model.predict_generator(validation_gen)
bottleneck_features_test = trimmed_model.predict_generator(test_gen)

np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)

