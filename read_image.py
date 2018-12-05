import glob
import os
import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
train_image_path = 'F:/github/tensorflow_alexnet_classify_myself6/TEST_ONE/'
image_path= np.array(glob.glob(train_image_path + '*.jpg')).tolist()
coll = io.ImageCollection(image_path)
print(len(coll))
for i in range(0,len(coll)):
    print (i+1)
    img=coll[i]
    plt.imshow(img)
    plt.show()