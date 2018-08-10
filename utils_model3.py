import cv2, os
import numpy as np
import matplotlib.image as mpimg
import random
import math

PROCESSED_IMG_COLS = 64
PROCESSED_IMG_ROWS = 64
PROCESSED_IMG_CHANNELS = 3

def crop_resize_image(img):
    shape = img.shape
    img = img[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    img = cv2.resize(img, (PROCESSED_IMG_COLS, PROCESSED_IMG_ROWS), interpolation=cv2.INTER_AREA)    
    return img
def preprocess(image):
    return crop_resize_image(image)
