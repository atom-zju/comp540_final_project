import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import os


Image_File = '/home/atom/cifar10/data/train/38124.png'

input_image = caffe.io.load_image(Image_File)
plt.imshow(input_image)
plt.show()
