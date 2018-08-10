from keras.models import load_model
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import cv2
import math

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

model = load_model('MODEL1.h5')
model.summary()

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

# choose a random number
i = np.random.randint(len(X_validation))
# choose a random center image from validation dataset
test_image = mpimg.imread(X_validation[i][0])

def resize(image):
    im =  cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    im = im.reshape(1,66,200,3)
    return im

test_image_resized = resize(test_image)

y_pred = model.predict(test_image_resized)

y_true = X_validation[i][3]

mse = (y_true - y_pred)**2

print('mse = ', mse)





y_pred = model.predict(test_image_resized)