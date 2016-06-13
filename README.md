# Caffe for remote sensing

This code extends Caffe framework for pixelwise labeling of aerial remote sensing imagery.

The main addition is a layer called "ImagePairDataLayer" (built upon ImageDataLayer) 
that simultaneously samples random patches from the images and the reference data.

In the folder "RemoteSensingBoston" we include the network configuration files to run the experiments presented in:

	@inproceedings{maggiori2016,
	  title={Fully Convolutional Neural Networks for Remote Sensing Image Classification},
	  author={Maggiori, Emmanuel and Tarabalka, Yuliya and Charpiat, Guillaume and Alliez, Pierre},
	  booktitle={Geoscience and Remote Sensing Symposium (IGARSS), 2016 IEEE International},
	  year={2016},
	  organization={IEEE}
	}
