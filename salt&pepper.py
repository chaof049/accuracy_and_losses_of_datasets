import cv2
import numpy as np
from skimage.util import random_noise

I = cv2.imread("/Users/supremepradhananga/Pycharm/custom/train/airplane/airplane5.jpg")
noise_img = random_noise(I, mode = 's&p', amount = 0.1)
noise_img = np.array(255*noise_img, dtype = 'uint8')
cv2.imwrite("/Users/supremepradhananga/Pycharm/custom/train/airplane/airplane6.jpg",noise_img)
cv2.imshow('blur', noise_img)
