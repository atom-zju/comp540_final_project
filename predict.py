import numpy as np
import matplotlib.pyplot as plt

import sys
import os

sys.path.append('/Users/atom/Downloads/caffe/python')
import caffe
caffe_root = '/home/atom/caffe/'
sys.path.insert(0,caffe_root + 'python')

Pretrained_File = '/home/atom/PycharmProjects/cifar10_2/cifar10_full_iter_60000.caffemodel.h5'
Model_File = '/home/atom/PycharmProjects/cifar10_2/cifar10_full.prototxt'
Image_Path = '/home/atom/cifar10/data/test/'

labels = [  'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck']

caffe.set_mode_cpu()
net = caffe.Classifier(Model_File, Pretrained_File,raw_scale=255, image_dims=(32,32), # channel_swap=(2,1,0),
                       mean=np.load(caffe_root + 'examples/cifar10/mean.cifar10.npy'))

# input_image = caffe.io.load_image(Image_File)
# plt.imshow(input_image)
# plt.show()

pred_file = open('/home/atom/PycharmProjects/cifar10_2/pred_file.csv','w+')
pred_file.write("id,label\n")
idx = 1
imageFileNames = os.listdir(Image_Path)
imageFileNames = sorted(imageFileNames, key=lambda x: int(x.split('.')[0]))

# for imageFileName in imageFileNames:
#     print imageFileName
# print "# of entries: ",len(imageFileNames)

for imageFileName in imageFileNames:
    input_image = caffe.io.load_image(Image_Path+imageFileName)
    prediction = net.predict([input_image])
    pred_file.write(str(idx)+","+labels[prediction[0].argmax()]+"\n")
    idx += 1
pred_file.close()