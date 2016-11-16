
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

#from numpy import unravel_index

from numpy import count_nonzero as nnz

import math

import sys
sys.path.append('path/CaffeRemoteSensing/python')
import caffe



def getimage(image_location,alpha_channel):
	im = Image.open(image_location)

	if (alpha_channel):
		im = im.convert("RGBA")

	in_ = np.array(im, dtype=np.float32)

	if alpha_channel:
		perm = [2,1,0,3]
		in_=in_[:,:,perm]

		#r,g,b,a=in_.T
		#in_ = np.array([b,g,r,a])
		#in_ = in_.transpose()
	else: 
		in_ = in_[:,:,::-1]

	#save in C-H-W order instead of H-W-C
	in_ = in_.transpose((2,0,1))


	return in_



def classify(net_config_location, net_weights, image_location, alpha_channel, use_gpu, win_size,crop_size):



	if use_gpu:
		caffe.set_device(0)
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()
	

	f = open(net_config_location, 'r')
	s = f.read()


	if (alpha_channel):
		append = 'input: "data" input_dim: 1 input_dim: 4 input_dim: 512 input_dim: 512 \n'
	else:
		append = 'input: "data" input_dim: 1 input_dim: 3 input_dim: 512 input_dim: 512 \n'	

	s = append + s

	f = open('temp.prototxt', 'w')
	f.write(s)
	f.close()


	net = caffe.Net('temp.prototxt', net_weights, caffe.TEST)

	print 'Opening image...'

	#open image
	in_ = getimage(image_location,alpha_channel)

	print 'Image opened...'

	#height and width of original image
	orig_img_h = in_.shape[1]
	orig_img_w = in_.shape[2]

	#size of valid output patch
	out_size = win_size-2*crop_size

	#number of patches horizontally and vertically
	n_patch_horiz = int(math.ceil(orig_img_w/float(out_size)))
	n_patch_vert = int(math.ceil(orig_img_h/float(out_size)))

	#pad image...

	#how much to pad?
	pad_w_before = crop_size
	pad_h_before = crop_size
	pad_w_after = n_patch_horiz*out_size + crop_size - orig_img_w
	pad_h_after = n_patch_vert*out_size + crop_size - orig_img_h

	#do padding
	in_ = np.pad(in_, ((0,0), (pad_h_before,pad_h_after), (pad_w_before,pad_w_after)), mode='symmetric')

	
	# shape for input (data blob is N x C x H x W), set data
	
	if alpha_channel:
		channels=4
	else:
		channels=3

	net.blobs['data'].reshape(1, channels, win_size, win_size)


	
	print 'Predicting...'


	rows = []

	for i in range(0,n_patch_vert):
		patches_in_row = []
		for j in range(0,n_patch_horiz):

			input_ = in_[:,out_size*i:out_size*i+win_size,out_size*j:out_size*j+win_size]
			net.blobs['data'].data[...] = input_

			# run net prediction
			net.forward()
			patch_out = net.blobs['prob'].data[0]

			
			#compute offset in case output patch provided by the network
			#is larger than it should be
			h_offset = (net.blobs['prob'].data[0].shape[1] - out_size)/2
			w_offset = (net.blobs['prob'].data[0].shape[2] - out_size)/2
			
			#crop
			patch_out = patch_out[:,h_offset:h_offset+out_size,w_offset:w_offset+out_size]

			patches_in_row.append(np.copy(patch_out))
		
		row = np.concatenate(patches_in_row,2)

		rows.append(np.copy(row))
	
	
	entire_output = np.concatenate(rows,1)

	#remove excess border
	output = entire_output[:,0:orig_img_h,0:orig_img_w]

	#out.astype('double').tofile("prob.dat");

	if output.shape[0]==1:
		pred = np.rint(np.squeeze(output)).astype(np.uint8)
	else:
		pred = output.argmax(axis=0).astype(np.uint8)

	print 'Done predicting.'


	return pred

