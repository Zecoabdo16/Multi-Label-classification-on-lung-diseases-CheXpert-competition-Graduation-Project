import cv2
import random

import pandas as pd
import numpy as np

import tensorflow as tf
import keras

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, main_dir, meta_data, n_classes, batch_size=32, shape=(224,224, 3), augment=True, val=False, shuffle=True):
        self.main_dir = main_dir
        self.meta_data = meta_data
        self.shape = shape
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.val = val
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.meta_data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(self.meta_data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def histogram_equalization(self, image):
        r,g,b = cv2.split(image)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        img = np.stack((r,g,b), -1)
        img = np.divide(img, 255)
        return img
    
    def brightness(self, img, value): 
        img = np.array(img, dtype = np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255] = 255
        hsv[:,:,2] = hsv[:,:,2]*value
        hsv[:,:,2][hsv[:,:,2]>255] = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def rotation(self, img, angle): 
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img

    def horizontal_flip(self, img):
        return cv2.flip(img, 1)

    def zoom(self, img, value): 
        h, w = img.shape[:2]
        h_start = int(value*h)
        w_start = int(value*w)
        h_end = h-h_start
        w_end = w-w_start
        img = img[h_start:h_end, w_start:w_end, :]
        img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
        return img
        
    def aug_func(self, image, brightness, rotation, horizontal_flip, zoom):
        if brightness:
          b = random.uniform(0.8, 1.3)
          image = self.brightness(image, b)

        if rotation:
          angle = int(random.uniform(-45, 45))
          image = self.rotation(image, angle)

        if horizontal_flip:
          image = self.horizontal_flip(image)

        if zoom:
          z = round(random.uniform(0.01, 0.15), 2)
          image = self.zoom(image, z)
        
        return image
          
    def __data_generation(self, indices):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, *self.shape))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, index in enumerate(indices):
            # Store sample
            image_id = self.meta_data.iloc[index, 0]
            image_path = image_id[:-9]+image_id[-4:] if not self.val else image_id

            image = cv2.imread(self.main_dir + image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.shape[0], self.shape[1]))
            image = self.histogram_equalization(image)

            # Augmentation
            if self.augment:
                b, r, h, z = map(int, list(image_id[-8:-4]))
                image = self.aug_func(image, b, r, h, z)
                
            X[i,] = image

            # Store class
            y[i,] = self.meta_data.iloc[index, 5:]

        return X, y