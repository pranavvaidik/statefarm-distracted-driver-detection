#import cv2
#import matplotlib.pyplot as plt
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True
import keras
import numpy as np
#define a function that reads a path to an image and returns a tensor suitable for keras


#Create Data Generator class that can generate data in batches and split them to train, test and validation sets
class DataGenerator(keras.utils.Sequence):
    
    #def __init__(self, list_file_paths, labels, batch_size=32, dim=(32,32,32), n_channels=1,
    #             n_classes=10, shuffle=True):
    def __init__(self, list_file_paths, labels, batch_size=32, shuffle=True):
        
        #Initialization
        #self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_file_paths = list_file_paths
        #self.n_channels = n_channels
        #self.n_classes = n_classes
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

