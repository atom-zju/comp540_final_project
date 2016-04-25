import caffe
import numpy as np
import skimage
import skimage.io
import matplotlib.pyplot as plt


Pretrained_Files = '/home/atom/cifar10/mymodel/rep2/_iter_70000.caffemodel.h5'
Model_File = '/home/atom/PycharmProjects/cifar10_2/cifar10_full.prototxt'
Mean_Files = '/home/atom/caffe/examples/cifar10/mean.cifar10.npy'

caffe.set_mode_cpu()
net = caffe.Classifier(Model_File, Pretrained_Files,raw_scale=255, image_dims=(32,32), # channel_swap=(2,1,0),
                       mean=np.load(Mean_Files))

w = net.params['conv1'][0].data
b = net.params['conv1'][1].data

weights = np.zeros((32,5,5,3),dtype=np.uint8)
for idx in range(32):
    for chn in range(3):
        tmp = w[idx,chn,:,:] + b[idx]
        min = np.min(tmp)
        max = np.max(tmp)
        weights[idx,:,:,chn] = 255.0 * (tmp - min) / (max - min)

for idx in range(32):
    plt.subplot(4,8,idx+1)
    plt.imshow(weights[idx,:,:,:])
    plt.axis('off')

plt.show()

