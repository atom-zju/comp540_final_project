import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import os
import csv
import utils

con_mat = utils.plot_confusion_matrix('pred_train_file.csv', '/home/atom/cifar10/data/trainLabels.csv')
print con_mat
np.save("confusion_matrix.txt", con_mat)

# with open("pred_file_rep.csv", 'r') as f:
#     reader = csv.reader(f)
#     my_list = list(reader)

# pred_file = open('pred_file_rep.csv','w+')
# pred_file.write("id,label\n")
#
# for idx in range(300000):
#     pred_file.write(str(idx+1)+","+my_list[idx+1][1]+"\n")
#
# pred_file.close()