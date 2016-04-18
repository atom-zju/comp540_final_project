import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
import os
import csv


with open("pred_file_rep.csv", 'r') as f:
    reader = csv.reader(f)
    my_list = list(reader)

# pred_file = open('pred_file_rep.csv','w+')
# pred_file.write("id,label\n")
#
# for idx in range(300000):
#     pred_file.write(str(idx+1)+","+my_list[idx+1][1]+"\n")
#
# pred_file.close()