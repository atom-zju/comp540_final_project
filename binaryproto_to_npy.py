import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/atom/cifar10/mydata/mean_rep_4.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( '/home/atom/cifar10/mymodel/rep4/mean.cifar10.npy', out)