import numpy as np
import skimage.io
import skimage.transform

def get_cifar10():
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

    cifar10_test_batch = unpickle(Batch_path+'test_batch')
    cifar10_test_data = cifar10_test_batch['data']
    cifar10_test_label = np.asarray(cifar10_test_batch['labels'])

    # print cifar10_batches[0]['data'].shape

    # stack the array all together
    cifar10_data = cifar10_batches[0]['data']
    cifar10_labels = cifar10_batches[0]['labels']

    for batch_idx in range(1,5):
        cifar10_data = np.vstack((cifar10_data,cifar10_batches[batch_idx]['data']))
        cifar10_labels += cifar10_batches[batch_idx]['labels']

    cifar10_labels = np.asarray(cifar10_labels)
    return cifar10_data, cifar10_labels, cifar10_test_data, cifar10_test_label

def mirror(image):
    total_cols = image.shape[1]
    mirror_img = np.copy(image)
    for idx in range(total_cols/2):
        mirror_img[:,total_cols-1-idx,:] = image[:,idx,:]
        mirror_img[:,idx,:] = image[:,total_cols-1-idx,:]
    return mirror_img


def rotate(image):
    # degree range of rotation
    rotate_range = 11
    rad = np.random.choice(rotate_range, 1) - rotate_range/2
    rotate_img = skimage.transform.rotate(image, rad,resize=False)
    if(rotate_img.shape != image.shape):
        print "shape incorrect in rotate"
    return rotate_img


def random_crop(image):
    # cropped size will have at least 80% of the original image
    margin_size = np.random.choice(int(image.shape[0] * 0.10), 1)
    cropped_size = image.shape[0]-margin_size
    vpos = np.random.choice(margin_size, 1)
    hpos = np.random.choice(margin_size, 1)
    cropped_img = np.copy(image[vpos:vpos+cropped_size,hpos:hpos+cropped_size,:])
    return skimage.transform.resize(cropped_img,image.shape)