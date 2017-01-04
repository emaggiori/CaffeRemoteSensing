


from classify_piecewise import classify




from PIL import Image
from os import listdir,path
import numpy as np

import time



def predict_folder(image_folder,output_prefix,
	       net_config_location,net_weights_location,
	       alpha_channel,use_gpu, win_size, crop_size):

	images = listdir(image_folder)

	
	#start counting time
	start_time = time.time()

	for i in range(0,len(images)):

		image_location = path.join(image_folder,images[i])
		pred = classify(net_config_location, net_weights_location, image_location, alpha_channel, use_gpu, win_size, crop_size)

		out = Image.fromarray(pred*255)
		out.save(output_prefix+images[i])


	#print elapsed time
	print("--- %s seconds ---" % (time.time() - start_time))









