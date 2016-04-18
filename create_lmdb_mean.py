import os

Image_Path = '/home/atom/cifar10/data/train/'
images_labels_path = '/home/atom/cifar10/mydata/'
Tool_Path = '/home/atom/caffe/tools/tools/'

os.system(Tool_Path+"convert_imageset --encode_type=\"png\" "+ Image_Path+' '+images_labels_path+'images_labels_rep_0.txt '
          +images_labels_path+'mycifar10_rep_0_lmdb')

os.system(Tool_Path+"convert_imageset --encode_type=\"png\" "+ Image_Path+' '+images_labels_path+'images_labels_rep_1.txt '
          +images_labels_path+'mycifar10_rep_1_lmdb')

os.system(Tool_Path+"convert_imageset --encode_type=\"png\" "+ Image_Path+' '+images_labels_path+'images_labels_rep_2.txt '
          +images_labels_path+'mycifar10_rep_2_lmdb')

os.system(Tool_Path+"convert_imageset --encode_type=\"png\" "+ Image_Path+' '+images_labels_path+'images_labels_rep_3.txt '
          +images_labels_path+'mycifar10_rep_3_lmdb')

os.system(Tool_Path+"convert_imageset --encode_type=\"png\" "+ Image_Path+' '+images_labels_path+'images_labels_rep_4.txt '
          +images_labels_path+'mycifar10_rep_4_lmdb')


os.system(Tool_Path+"compute_image_mean --backend=\"lmdb\" "+images_labels_path+'mycifar10_rep_0_lmdb '
          +images_labels_path+'mean_rep_0.binaryproto')

os.system(Tool_Path+"compute_image_mean --backend=\"lmdb\" "+images_labels_path+'mycifar10_rep_1_lmdb '
          +images_labels_path+'mean_rep_1.binaryproto')

os.system(Tool_Path+"compute_image_mean --backend=\"lmdb\" "+images_labels_path+'mycifar10_rep_2_lmdb '
          +images_labels_path+'mean_rep_2.binaryproto')

os.system(Tool_Path+"compute_image_mean --backend=\"lmdb\" "+images_labels_path+'mycifar10_rep_3_lmdb '
          +images_labels_path+'mean_rep_3.binaryproto')

os.system(Tool_Path+"compute_image_mean --backend=\"lmdb\" "+images_labels_path+'mycifar10_rep_4_lmdb '
          +images_labels_path+'mean_rep_4.binaryproto')