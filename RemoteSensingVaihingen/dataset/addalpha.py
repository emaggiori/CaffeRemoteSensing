from PIL import Image
import numpy as np
from os import listdir,path

def addalpha(image_location,alpha_location):

	img = Image.open(image_location)
	alpha = Image.open(alpha_location)
	print img.size
	print alpha.size
	img.putalpha(alpha)

	return img




input_directory = '/user/emaggior/home/Documents/Images/isprs/ISPRS_semantic_labeling_Vaihingen/top'
alpha_directory = '/user/emaggior/home/Documents/Images/isprs/normalized_DSM'
output_directory = 'alpha'

files = listdir(input_directory)

indices=[1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,37,38]
for i in indices:
	image_location = path.join(input_directory,'top_mosaic_09cm_area'+str(i)+'.tif')
	alpha_location = path.join(alpha_directory,'dsm_09cm_matching_area'+str(i)+'_normalized.jpg')
	added = addalpha(image_location,alpha_location)
	save_to = path.join(output_directory,'topdsm'+str(i)+'.png')
	added.save(save_to)






