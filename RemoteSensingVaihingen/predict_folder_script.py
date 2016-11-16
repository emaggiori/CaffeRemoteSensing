

from predict_folder import predict_folder


from PIL import Image
from os import listdir,path
import numpy as np


net_weights_location = "path/snap/mlp/_iter_30000.caffemodel"

net_config_location = "path/config/mlp.prototxt"

image_folder = 'path/dataset_alpha/val'
#image_folder = 'path/dataset_alpha/test'

ref_folder = 'path/converted_gt/valeroded'
#ref_folder = ''

output_prefix = 'outputPrefix/'


#use_gpu=0
use_gpu=1
alpha_channel = 1

win_size = 1600
crop_size = 37




out_acc =[]

accuracies =  predict_folder(image_folder,ref_folder,output_prefix,
			     net_config_location,net_weights_location,
			     alpha_channel,use_gpu,win_size,crop_size)

if (ref_folder!=''):

	print accuracies
	
	out_acc.append(accuracies['acc'])
	out_acc.append(accuracies['mean_f1'])
	out_acc.append(accuracies['meanIou'])
	out_acc.extend(accuracies['f1'])
	out_acc.extend(accuracies['ious'])

	np_array = np.array(out_acc)
	np.savetxt(output_prefix+'acc.txt',np_array)





