

from predict_folder import predict_folder


from PIL import Image
from os import listdir,path
import numpy as np



net_weights_location = "trainedModels/mlpModel.caffemodel"

net_config_location = "config/mlp.prototxt"

#images to predict
image_folder = 'benchmark/test/images'

output_prefix = 'predictions/'



use_gpu=1
alpha_channel = 0

win_size = 1024
crop_size = 74

predict_folder(image_folder,output_prefix,
               net_config_location,net_weights_location,
	       alpha_channel,use_gpu,win_size,crop_size)




