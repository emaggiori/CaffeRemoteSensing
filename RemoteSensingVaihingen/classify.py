
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

#from numpy import unravel_index

from numpy import count_nonzero as nnz

import math

import sys
sys.path.append('path/CaffeRemoteSensing/python')
import caffe



def classify(net_config_location, net_weights, image_location, alpha_channel, use_gpu):


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


	im = Image.open(image_location)

	if (alpha_channel):
		im = im.convert("RGBA")

	in_ = np.array(im, dtype=np.float32)


	if alpha_channel:
		r,g,b,a=in_.T
		in_ = np.array([b,g,r,a])
		in_ = in_.transpose()
	else: 
		in_ = in_[:,:,::-1]

	#in_-=127

	#vis = Image.fromarray(in_.astype(np.uint8))
	#vis.show()

	#save in C-H-W order instead of H-W-C
	in_ = in_.transpose((2,0,1))



	#height and width of original image
	h = in_.shape[1]
	w = in_.shape[2]


	#pad image

	padded_h = int(math.ceil(h/16.0))*16
	padded_w = int(math.ceil(w/16.0))*16

	pad_h = padded_h - h
	pad_w = padded_w - w

	in_ = np.pad(in_, ((0,0), (0,pad_h), (0,pad_w)), mode='symmetric')


	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_


	# run net and take argmax for prediction
	net.forward()

	out = net.blobs['prob'].data[0]
	out_sub = out[:,0:h,0:w]
	#out_sub = out

	#out.astype('double').tofile("prob.dat");

	if out_sub.shape[0]==1:
		pred = np.rint(np.squeeze(out_sub)).astype(np.uint8)
	else:
		pred = out_sub.argmax(axis=0).astype(np.uint8)


	#imgplot = plt.imshow(pred)
	#plt.colorbar()
	#plt.show()

	#oneclass = net.blobs['prob'].data[0][3]
	#imgplot = plt.imshow(oneclass)
	#plt.colorbar()
	#plt.show()



	return pred

