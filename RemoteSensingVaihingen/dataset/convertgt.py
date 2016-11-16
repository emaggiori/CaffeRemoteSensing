from PIL import Image
import numpy as np
from os import listdir,path

def convertimage(image_location):

	img = Image.open(image_location)

	arr = np.array(img)

	r = arr[:,:,0]
	g = arr[:,:,1]
	b = arr[:,:,2]


	output = np.zeros(r.shape, dtype=np.uint8)



	mask0 = (r==255) & (g==255) & (b==255)
	mask1 = (r==0) & (g==0) & (b==255)
	mask2 = (r==0) & (g==255) & (b==255)
	mask3 = (r==0) & (g==255) & (b==0)
	mask4 = (r==255) & (g==255) & (b==0)
	mask5 = (r==255) & (g==0) & (b==0)
	mask6 = (r==0) & (g==0) & (b==0)



	output[mask0] = 0
	output[mask1] = 40
	output[mask2] = 80
	output[mask3] = 120
	output[mask4] = 160
	output[mask5] = 200
	output[mask6] = 240

	img = Image.fromarray(output)
	
	return img




input_directory = '/user/emaggior/home/Documents/Images/isprs/ISPRS_semantic_labeling_Vaihingen/gts_for_participants'
#input_directory = '/user/emaggior/home/Documents/Images/isprs/ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants'
output_directory = 'converted_gt'

files = listdir(input_directory)

for f in files:
	image_location = path.join(input_directory,f)
	converted = convertimage(image_location)
	save_to = path.join(output_directory,f)
	converted.save(save_to)





