
import numpy as np
import sys
sys.path.append('/Users/atom/Downloads/caffe/python/')
import caffe
import lmdb
import skimage.io
import skimage.transform

# note that in skimage, the image format is  RGB-image MxNx3 for color image, and for lmdb the format is 3*MxN


from utils import rotate
from utils import mirror
from utils import random_crop

def trans_lmdb2ski(image):
    # from 3*MxN to MxN*3
    result = np.zeros((image.shape[1],image.shape[2], image.shape[0]),dtype=np.uint8)
    for chn in range(3):
        result[:,:,chn] = image[chn,:,:]
    return result

def trans_ski2lmdb(image):
    # from MxN*3 to 3*MxN
    result = np.zeros((image.shape[2],image.shape[0], image.shape[1]),dtype=np.uint8)
    for chn in range(3):
        result[chn,:,:] = image[:,:,chn]
    return  result

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

Batch_path = '/Users/atom/cifar10/cifar-10-batches-py/'
# load the python version of cifar 10
cifar10_batches = []
cifar10_batches.append(unpickle(Batch_path+'data_batch_1'))
cifar10_batches.append(unpickle(Batch_path+'data_batch_2'))
cifar10_batches.append(unpickle(Batch_path+'data_batch_3'))
cifar10_batches.append(unpickle(Batch_path+'data_batch_4'))
cifar10_batches.append(unpickle(Batch_path+'data_batch_5'))

# print cifar10_batches[0]['data']

# stack the array all together
cifar10_data = cifar10_batches[0]['data']
cifar10_labels = cifar10_batches[0]['labels']

for batch_idx in range(1,5):
    cifar10_data = np.vstack((cifar10_data,cifar10_batches[batch_idx]['data']))
    cifar10_labels += cifar10_batches[batch_idx]['labels']

N = cifar10_data.shape[0]
X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
#X_enlarged = np.zeros((N, 3, 32*7, 32*7), dtype=np.uint8)

for img_idx in range(N):
    for chn in range(3):
        X[img_idx,chn,:,:] = np.reshape(cifar10_data[img_idx,chn*1024:(chn+1)*1024],(32,32))
        #X_enlarged[img_idx,chn,:,:] = np.repeat(np.repeat(X[img_idx,chn,:,:],7,axis=0),7,axis=1)

y = cifar10_labels

#X = X_enlarged

# print X.shape
# print len(cifar10_labels)

# do data augmentation here

# augmented image numbers will define the fraction being augment
augment_num = int(0.5*N)

verbos = False

mirror_idxs = np.random.choice(N, augment_num, replace=False)
crop_idxs = np.random.choice(N, augment_num, replace=False)
rotate_idxs = np.random.choice(N, augment_num, replace=False)

# do the mirror augmentation here
aux_array = np.zeros((augment_num,X.shape[1],X.shape[2],X.shape[3]),dtype=np.uint8)
for cnt, aux_idx in enumerate(mirror_idxs):
    trans_img = trans_lmdb2ski(X[aux_idx,:,:,:])
    aug_img = mirror(trans_img)
    if cnt%1000 ==0 and verbos:
        skimage.io.imshow(aug_img)
        skimage.io.show()
        raw_input("hit enter to continue")
    re_trans_img = trans_ski2lmdb(aug_img)
    aux_array[cnt,:,:,:] = re_trans_img
    y.append(y[aux_idx])

print X.shape
print aux_array.shape

X = np.append(X,aux_array,axis=0)
N = X.shape[0]

# # do the crop augmentation here
# aux_array = np.zeros((augment_num,X.shape[1],X.shape[2],X.shape[3]),dtype=np.uint8)
# for cnt, aux_idx in enumerate(crop_idxs):
#     trans_img = trans_lmdb2ski(X[aux_idx,:,:,:])
#     aug_img = random_crop(trans_img)
#     if cnt%1000 ==0 and verbos:
#         skimage.io.imshow(aug_img)
#         skimage.io.show()
#         raw_input("hit enter to continue")
#     re_trans_img = trans_ski2lmdb(aug_img)
#     aux_array[cnt,:,:,:] = re_trans_img
#     y.append(y[aux_idx])
#
# X = np.append(X,aux_array,axis=0)
# N = X.shape[0]
#
# # do the rotation augmentation here
# aux_array = np.zeros((augment_num,X.shape[1],X.shape[2],X.shape[3]),dtype=np.uint8)
# for cnt, aux_idx in enumerate(rotate_idxs):
#     trans_img = trans_lmdb2ski(X[aux_idx,:,:,:])
#     aug_img = rotate(trans_img)
#     if cnt%1000 ==0 and verbos:
#         skimage.io.imshow(aug_img)
#         skimage.io.show()
#         raw_input("hit enter to continue")
#     re_trans_img = trans_ski2lmdb(aug_img)
#     aux_array[cnt,:,:,:] = re_trans_img
#     y.append(y[aux_idx])
#
# X = np.append(X,aux_array,axis=0)
N = X.shape[0]

# here we begin to create lmdb file
#
# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes * 10

env = lmdb.open('mylmdb2', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)
        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())



