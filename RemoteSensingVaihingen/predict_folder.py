

from acc import acc
from classify_piecewise import classify
#from classify import classify


from PIL import Image
from os import listdir,path
import numpy as np

import time


def predict_folder(image_folder,ref_folder,output_prefix,
	       net_config_location,net_weights_location,
	       alpha_channel,use_gpu, win_size, crop_size):

	images = listdir(image_folder)
	

	if (ref_folder!=''):
		pred_concat = np.array([])
		ref_concat = np.array([])
		refs = listdir(ref_folder)

	#start counting time
	start_time = time.time()

	for i in range(0,len(images)):

		image_location = path.join(image_folder,images[i])
		pred = classify(net_config_location, net_weights_location, image_location, alpha_channel, use_gpu, win_size, crop_size)

		out = Image.fromarray(pred*40)
		out.save(output_prefix+images[i])

		if (ref_folder!=''):
			ref_location = path.join(ref_folder,refs[i])
			ref = np.array(Image.open(ref_location))/40

			pred=pred.flatten()
			ref=ref.flatten()
	
			pred_concat=np.concatenate((pred_concat,pred))
			ref_concat=np.concatenate((ref_concat,ref))


	#print elapsed time
	print("--- %s seconds ---" % (time.time() - start_time))

	if (ref_folder!=''):
		accuracies = acc(pred_concat,ref_concat)
		return accuracies
	else:
		return 0








