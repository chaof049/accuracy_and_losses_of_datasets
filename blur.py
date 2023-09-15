import cv2
from matplotlib import pyplot as plt
from skimage.util import random_noise

I = cv2.imread("/Users/supremepradhananga/Pycharm/custom//train/airplane/airplane1.jpg")
rows, cols = I.shape[:2]

gauss = random_noise(I, mode = 'gaussian', seed = None, clip = True)
sp = random_noise(I, mode = 's&p', seed = None, clip = True)
salt = random_noise(I, mode = 'salt', seed = None, clip = True)
pepper = random_noise(I, mode = 'pepper', seed = None, clip = True)

plt.subplot(241), plt.imshow(I), plt.title('origin')
plt.subplot(242), plt.imshow(gauss), plt.title('gaussian')
plt.subplot(243), plt.imshow(sp), plt.title('salt&pepper')
plt.subplot(244), plt.imshow(salt), plt.title('salt')
plt.subplot(245), plt.imshow(pepper), plt.title('pepper')
plt.show();