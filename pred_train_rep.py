import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import os
import csv

caffe_root = '/home/atom/caffe/'
sys.path.insert(0,caffe_root + 'python')

Pretrained_Files = []
Pretrained_Files.append('/home/atom/cifar10/mymodel/rep2/_iter_70000.caffemodel.h5')
Pretrained_Files.append('/home/atom/cifar10/mymodel/rep3/_iter_70000.caffemodel.h5')
Pretrained_Files.append('/home/atom/cifar10/mymodel/rep0/_iter_70000.caffemodel.h5')
Pretrained_Files.append('/home/atom/cifar10/mymodel/rep4/_iter_70000.caffemodel.h5')
Pretrained_Files.append('/home/atom/cifar10/mymodel/rep1/_iter_70000.caffemodel.h5')
Pretrained_Files.append('/home/atom/cifar10/mymodel/_iter_70000.caffemodel.h5')
Pretrained_Files.append('/home/atom/caffe/examples/cifar10/cifar10_full_iter_70000.caffemodel.h5')

Pretrained_Files.append('/home/atom/caffe/examples/cifar10/cifar10_full_iter_65000.caffemodel.h5')
Pretrained_Files.append('/home/atom/PycharmProjects/cifar10_2/cifar10_full_iter_60000.caffemodel.h5')

Model_File = '/home/atom/PycharmProjects/cifar10_2/cifar10_full.prototxt'
Image_Path = '/home/atom/cifar10/data/train/'

# define the mean file paths
Mean_Files = []
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/caffe/examples/cifar10/mean.cifar10.npy')
Mean_Files.append('/home/atom/cifar10/mymodel/rep4/mean.cifar10.npy')

Label_File = '/home/atom/cifar10/data/trainLabels.csv'

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
nets = []
for idx in range(7):
    net = caffe.Classifier(Model_File, Pretrained_Files[idx],raw_scale=255, image_dims=(32,32), #channel_swap=(2,1,0),
                       mean=np.load(Mean_Files[idx]))
    nets.append(net)

# input_image = caffe.io.load_image(Image_File)
# plt.imshow(input_image)
# plt.show()

with open(Label_File, 'r') as f:
    reader = csv.reader(f)
    my_list = list(reader)

# print my_list

pred_file = open('pred_train_file.csv','w+')
pred_file.write("id,label\n")
idx = 1
imageFileNames = os.listdir(Image_Path)
imageFileNames = sorted(imageFileNames, key=lambda x: int(x.split('.')[0]))

# for imageFileName in imageFileNames:
#     print imageFileName
# print "# of entries: ",len(imageFileNames)

correct_cnt = 0

for imageFileName in imageFileNames:
    #print "predicting image: "+imageFileName
    input_image = caffe.io.load_image(Image_Path+imageFileName)
    pred_rep = np.zeros(10)
    for rep_idx in range(7):
        prediction = nets[rep_idx].predict([input_image])
        #print "expert "+str(rep_idx)+" predicts: "+str(prediction[0].argmax())
        pred_rep[prediction[0].argmax()] += 1
    pred_file.write(str(idx)+","+labels[pred_rep.argmax()]+"\n")
    pred_label = labels[pred_rep.argmax()]
    actual_label = my_list[idx][1]
    if pred_label == actual_label:
        correct_cnt+=1
    if idx%100==0:
        print "accuraccy on first ",idx," example is ", float(correct_cnt)/float(idx)
    idx += 1
pred_file.close()

print "model accuraccy on training data is ", float(correct_cnt)/len(imageFileNames)