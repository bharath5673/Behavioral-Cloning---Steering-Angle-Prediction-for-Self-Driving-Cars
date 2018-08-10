import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import json
import math

import keras
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import threading
import tensorflow as tf
from IPython.display import display
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import h5py
from keras.models import load_model

# fix random seed reproducibility
# This helps in debugging
seed = 7
np.random.seed(seed)

columns = ['center', 'left', 'right', 'steering_angle', 'throttle',
           'brake', 'speed']
# Loading Data
print('Loading Dataset ...')
filepath = os.path.join(os.getcwd(), 'driving_log.csv')
data = pd.read_csv(filepath, names = columns)

# Data description
print('Dataset_Columns: ', columns, '\n')
print('Dataset shape: ', data.shape, '\n')
print(data.describe(), '\n')

print('Dataset Loaded...')

# Exploring dataset
# Histogram of Steering Angles before Image Augmentation
binwidth = 0.025
plt.hist(data.steering_angle, 
         bins = np.arange(min(data.steering_angle), 
                                               max(data.steering_angle) 
                                               + binwidth, binwidth))
plt.title('Number of images per steering angle')
plt.xlabel('Steering Angle')
plt.ylabel('# Frames')
plt.show()      

# Train and Validation split data in 90 : 10 ratio

# shuffle data
data = data.reindex(np.random.permutation(data.index))

num_train = int((len(data) / 10.) * 9.) 

# Slicing the dataframe
X_train = data.iloc[:num_train]
X_validation = data.iloc[num_train:]

print('X_train has {} elements', format(len(X_train))) 
print('X_validation has {} elements', format(len(X_validation))) 

# Image Augmentation and Pre-processing Hyper Parameters
CAMERA_OFFSET = 0.25
CHANNEL_SHIFT_RANGE = 0.2
WIDTH_SHIFT_RANGE = 100
HEIGHT_SHIFT_RANGE = 40
PROCESSED_IMG_COLS = 64
PROCESSED_IMG_ROWS = 64
PROCESSED_IMG_CHANNELS = 3

# Model Hyper Parameters
#NB_EPOCH = 20
NB_EPOCH = 10

BATCH_SIZE = 64

# Data Augmentation Functions

# flip images horizontally
def horizontal_flip(img, steering_angle):
    flipped_image = cv2.flip(img, 1)
    steering_angle = -1 * steering_angle
    return flipped_image, steering_angle
# shift range for each channels
def channel_shift(img, channel_shift_range = CHANNEL_SHIFT_RANGE):
    img_channel_index = 2 #tf indexing
    channel_shifted_image = random_channel_shift(img, channel_shift_range,
                                                 img_channel_index)
    return channel_shifted_image
# shift height/width of the image by a small amount
def height_width_shift(img, steering_angle):
    rows, cols, channels = img.shape
    
    #Translation
    tx = WIDTH_SHIFT_RANGE * np.random.uniform() - WIDTH_SHIFT_RANGE / 2
    ty = HEIGHT_SHIFT_RANGE * np.random.uniform() - HEIGHT_SHIFT_RANGE / 2
    steering_angle = steering_angle + tx / WIDTH_SHIFT_RANGE * 2 * 0.2
    
    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])
    
    translated_image = cv2.warpAffine(img, transform_matrix, (cols, rows))
    return translated_image, steering_angle
# brightness shift
def brightness_shift(img, bright_value = None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if bright_value:
        img[:,:,2] += bright_value
    else:
        random_bright = 0.25 + np.random.uniform()
        img[:,:,2] = img[:,:,2] * random_bright
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img
# crop the image
def crop_resize_image(img):
    shape = img.shape
    img = img[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    img = cv2.resize(img, (PROCESSED_IMG_COLS, PROCESSED_IMG_ROWS), interpolation=cv2.INTER_AREA)    
    return img
# Combining Augmentations
    # Wrapper Function
def apply_random_transformation(img, steering_angle):
    
    transformed_image, steering_angle = height_width_shift(img, steering_angle)
    transformed_image = brightness_shift(transformed_image)
    # transformed_image = channel_shift(transformed_image) # increasing train time. not much benefit. commented
    
    if np.random.random() < 0.5:
        transformed_image, steering_angle = horizontal_flip(transformed_image, steering_angle)
            
    transformed_image = crop_resize_image(transformed_image)
    
    return transformed_image, steering_angle

# Image Augmentation Visualization
test_row = data.values[np.random.randint(len(data.values))]
test_img = cv2.imread(test_row[0])
test_steer = test_row[3]
def aug_visualization(test_img, test_steer):
    #original image
    plt.figure()
    plt.xlabel('Original Test Image, Steering angle :' + str(test_steer))
    plt.imshow(test_img)
    #horizontally flipped image
    flipped_image, new_steering_angle = horizontal_flip(test_img, test_steer)
    plt.figure()
    plt.xlabel("Horizontally Flipped, New steering angle: " + str(new_steering_angle))
    plt.imshow(flipped_image)
    #channel shifted image
    channel_shifted_image = channel_shift(test_img, 255)
    plt.figure()
    plt.xlabel("Random Channel Shifted, Steering angle: " + str(test_steer))
    plt.imshow(channel_shifted_image)
    # width shifted image
    width_shifted_image, new_steering_angle = height_width_shift(test_img, test_steer)
    new_steering_angle = "{:.7f}".format(new_steering_angle)
    plt.figure()
    plt.xlabel("Random HT and WD Shifted, New steering angle: " + str(new_steering_angle))
    plt.imshow(width_shifted_image)
    #brightened image
    brightened_image = brightness_shift(test_img, 255)
    plt.figure()
    plt.xlabel("Brightened, Steering angle: " + str(test_steer))
    plt.imshow(brightened_image)
    #crop 
    cropped_image = crop_resize_image(test_img)
    plt.figure()
    plt.xlabel("Cropped and Resized, Steering angle: " + str(test_steer))
    plt.imshow(cropped_image)

def load_and_augment_image(line_data):
    i = np.random.randint(3)
    
    if (i == 0):
        path_file = line_data['left'][0].strip()
        shift_angle = CAMERA_OFFSET
    elif (i == 1):
        path_file = line_data['center'][0].strip()
        shift_angle = 0
    elif (i == 2):
        path_file = line_data['right'][0].strip()
        shift_angle = - 1 * CAMERA_OFFSET
    steering_angle = line_data['steering_angle'][0] + shift_angle
    
    img = cv2.imread(path_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, steering_angle = apply_random_transformation(img, steering_angle)
    return img, steering_angle


generated_steering_angles = []
threshold = 1

#@threadsafe_generator
def generate_batch_data(_data, batch_size = 32):
    
    batch_images = np.zeros((batch_size, PROCESSED_IMG_ROWS, PROCESSED_IMG_COLS, PROCESSED_IMG_CHANNELS))
    batch_steering = np.zeros(batch_size)
    
    while 1:
        for batch_index in range(batch_size):
            row_index = np.random.randint(len(_data))
            line_data = _data.iloc[[row_index]].reset_index()
            
            # idea borrowed from Vivek Yadav: Sample images such that images with lower angles 
            # have lower probability of getting represented in the dataset. This alleviates 
            # any problems we may encounter due to model having a bias towards driving straight.
            
            keep = 0
            while keep == 0:
                x, y = load_and_augment_image(line_data)
                if abs(y) < .1:
                    val = np.random.uniform()
                    if val > threshold:
                        keep = 1
                else:
                    keep = 1
            
            batch_images[batch_index] = x
            batch_steering[batch_index] = y
            generated_steering_angles.append(y)
        yield batch_images, batch_steering

iterator = generate_batch_data(X_train, batch_size=10)
sample_images, sample_steerings = iterator.__next__()
def batch_generation_visualization():
    
    plt.subplots(figsize=(20, 5))
    for i, img in enumerate(sample_images):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.title("Steering: {:.4f}".format(sample_steerings[i]))
        plt.imshow(img)
    plt.show()


model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(PROCESSED_IMG_ROWS, PROCESSED_IMG_COLS, PROCESSED_IMG_CHANNELS)))
model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2), name='Conv1'))
model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2), name='Conv2'))
model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
#model = load_model('modelv2-3.h5')
model.summary()

#model = load_model('model-1.h5')
#model = load_model('model-2.h5')

# checkpoint
checkpoint = ModelCheckpoint('modelv2-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
# compile
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse', metrics=[])

class LifecycleCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global threshold
        threshold = 1 / (epoch + 1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        print('BEGIN TRAINING')
        self.losses = []

    def on_train_end(self, logs={}):
        print('END TRAINING')

# Calculate the correct number of samples per epoch based on batch size
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    samples = math.ceil(num_batches)
    samples_per_epoch = samples * batch_size
    return samples_per_epoch

lifecycle_callback = LifecycleCallback()       

train_generator = generate_batch_data(X_train, BATCH_SIZE)
validation_generator = generate_batch_data(X_validation, BATCH_SIZE)

samples_per_epoch = calc_samples_per_epoch((len(X_train)*3), BATCH_SIZE)
nb_val_samples = calc_samples_per_epoch((len(X_validation)*3), BATCH_SIZE)

history = model.fit_generator(train_generator, 
                              validation_data = validation_generator,
                              samples_per_epoch = len(X_train), 
                              nb_val_samples = len(X_validation),
                              nb_epoch = NB_EPOCH, 
                              verbose=1,
                              callbacks=[lifecycle_callback, checkpoint])
model.save('./modelv2-4.h5')
model_json = model.to_json()
with open("./modelv2-004.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./modelv2-004.h5")
print("Saved model to disk")

# list all data in history
print(history.history.keys())

# summarize history for epoch loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.figure()
plt.hist(generated_steering_angles, bins=np.arange(min(generated_steering_angles), max(generated_steering_angles) + binwidth, binwidth))
plt.title('Number of augmented images per steering angle')
plt.xlabel('Steering Angle')
plt.ylabel('# Augmented Images')
plt.show()

# summarize history for batch loss
plt.figure()
batch_history = lifecycle_callback.losses
plt.plot(batch_history)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('batches')
plt.show()

# Layer visualizations

test_fn = "IMG/center_2016_12_01_13_32_43_457.jpg"

def visualize_model_layer_output(layer_name):
    model2 = Model(input=model.input, output=model.get_layer(layer_name).output)

    img = load_img(test_fn)
    img = crop_resize_image(img_to_array(img))
    img = np.expand_dims(img, axis=0)

    conv_features = model2.predict(img)
    print("conv features shape: ", conv_features.shape)
    
    # plot features
    plt.subplots(figsize=(5, 5))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(conv_features[0,:,:,i], cmap='gray')
    plt.show()
visualize_model_layer_output('Conv1')
visualize_model_layer_output('Conv2')