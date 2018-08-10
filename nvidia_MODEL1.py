import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
from keras.preprocessing.image import *
import random

# Generates the same set of random numbers everytime,
# which makes it deterministic for debugging
np.random.seed(0)

# nvidia model parameters
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Loading dataset as a pandas data frame

filepath = os.path.join(os.getcwd(), 'driving_log.csv')
print('Loading dataset ...')
    
dataFrame = pd.read_csv(filepath, names = ['center','left', 'right',
                                               'steering', 'throttle', 'brake',
                                               'speed'])
    
Xdata = dataFrame[['center', 'left', 'right']].values
ydata = dataFrame['steering'].values
data_samples = dataFrame.values
X_train, X_validation = train_test_split(data_samples, test_size = 0.2)
    
print('Dataset has been Loaded ...')
    


# Data Augmentations

# This function is used to change brightness of the images
def brightness_shift(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    image_hsv[:,:,2] = image_hsv[:,:,2] * ratio
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    return image

# This function is used to generate random shadows
def generate_shadow(image):
    bright_factor = 0.3
    
    x = random.randint(0, image.shape[1])
    y = random.randint(0, image.shape[0])

    width = random.randint(int(image.shape[1]/2),image.shape[1])
    if(x+ width > image.shape[1]):
        x = image.shape[1] - x
    height = random.randint(int(image.shape[0]/2),image.shape[0])
    if(y + height > image.shape[0]):
        y = image.shape[0] - y
    
    #Assuming HSV image
    image[y:y+height,x:x+width,:] = image[y:y+height,x:x+width,:]*bright_factor

    return image

# This function is used to translate the image by random amounts
def random_translate(image, steering_angle, range_x = 50, range_y = 5):
    steering_angle = float(steering_angle)
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

# This function is used to flip the image in the hozizontal axis
def flip(image, steering_angle):
    steering_angle = float(steering_angle)
    image = cv2.flip(image, 1)
    steering_angle = -1 * steering_angle
    return image, steering_angle

# This function is used to crop the image to remove all 
    # other elements except road.
def crop(image):
    return image[60:-25, :, :]

# This function is used to resize the image to nvidia model specifications
def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

# This image converts RGB images to YUV 
    # This helps in reducing errors while image capturing
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# This function combines all the preprocessing steps
def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

# This function combines all augmentations
def augmentImage(image, steering_angle):
   
    steering_angle = float(steering_angle)
    if np.random.random() <= 0.8:
        image, steering_angle = flip(image, steering_angle)
        image, steering_angle = random_translate(image, steering_angle)
    if np.random.random() <= 0.5:
        image = brightness_shift(image)
        image = generate_shadow(image)
    
    return image, steering_angle

# This function is used to generate image batches
def batch_generator(data_samples, batch_size = 50):
    while True:
        random.shuffle(data_samples)
        num_samples = len(data_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data_samples[offset : offset + 50]
            image_data = []
            steers = []
            for row in batch_samples:
                x_center = cv2.imread(row[0])
                x_left = cv2.imread(row[1])
                x_right = cv2.imread(row[2])
                y_center = float(row[3])
                y_left = float(row[3]) + 0.2
                y_right = float(row[3]) - 0.2
                
                x_center, y_center = augmentImage(x_center, y_center)
                x_left, y_left = augmentImage(x_left, y_left)
                x_right, y_right = augmentImage(x_right, y_right)
                x_center = preprocess(x_center)
                x_left = preprocess(x_left)
                x_right = preprocess(x_right)
                image_data.append(x_center)
                image_data.append(x_left)
                image_data.append(x_right)
                steers.append(y_center)
                steers.append(y_left)
                steers.append(y_right)
                
            image_data, steers = sklearn.utils.shuffle(np.array(image_data), np.array(steers))
            yield image_data, steers

# nvidia model
'''
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
'''                
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))            
        
model.fit_generator(batch_generator(X_train),
                        validation_data=batch_generator(X_validation),
                        nb_val_samples=len(X_validation),
                        samples_per_epoch = len(X_train), nb_epoch = 10,
                        verbose = 1)


model.save('./model1.h5')
    
    
        

    



    
    
    

