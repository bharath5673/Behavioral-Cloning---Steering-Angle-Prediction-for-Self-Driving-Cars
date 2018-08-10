import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import cv2
import math
from keras.models import model_from_json
from keras.models import load_model

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 224, 224, 3

model = load_model('MODEL6.h5')
model.summary()

# loading dataset
df = pd.read_csv('interpolated.csv')
df.drop(['index','timestamp','width','height',
         'lat','long','alt'], 1, inplace = True)
data = df.values
# center, left and right cameras
left = []
left_steer = []
right = []
right_steer = []
center = []
center_steer = []
for i in range(len(df)):
    if df['frame_id'][i] == 'left_camera':
        left.append(df['filename'][i])
        left_steer.append(df['angle'][i])
    elif df['frame_id'][i] == 'right_camera':
        right.append(df['filename'][i])
        right_steer.append(df['angle'][i])
    elif df['frame_id'][i] == 'center_camera':
        center.append(df['filename'][i])
        center_steer.append(df['angle'][i])
        
# choose a random number
i = np.random.randint(len(center))
# choose a random center image from validation dataset
test_image = mpimg.imread(center[i])

def resize(image):
    im =  cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    im = im.reshape(1,224,224,3)
    return im

test_image_resized = resize(test_image)
y_true = center_steer[i]

y_pred = model.predict(test_image_resized)


mse = (y_true - y_pred)**2

print('mse = ', mse)