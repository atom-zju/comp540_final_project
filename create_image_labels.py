'''
this piece of code will generate images_labels.txt in the form of "XXX.png L" (L is the label)
after creating these files, I ran the command in the tool directory of caffe, which will create a lmdb file for you

./convert_imageset --encode_type="png" /home/atom/cifar10/data/train/ /home/atom/cifar10/mydata/images_labels_rep_0.txt /home/atom/cifar10/mydata/mycifar10_rep_0_lmdb

./compute_image_mean -backend=lmdb /home/atom/cifar10/mydata/mycifar10_rep_0_lmdb /home/atom/cifar10/mydata/mean_rep_0.binaryproto
'''
import numpy as np

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

csv = np.genfromtxt('trainLabels2.csv',dtype=None, delimiter=",")
print csv.shape[0]

image_idx_file = open('images_labels.txt','w+')

for idx in xrange(csv.shape[0]):
    # do the conversion
    image_idx_file.write(str(csv[idx][0])+'.png '+str(labels.index(csv[idx][1]))+"\n")
image_idx_file.close()
