# Networks and trained models for Inria Aerial Image Labeling Dataset


To predict, put all the images in a single folder and call predict_folder_script.py (after setting the required parameters in the script).


To train the FCN:

./build/tools/caffe train -solver=./config/fcn_solver.prototxt

To train the MLP from the pretrained FCN weights:

./build/tools/caffe train -solver=./config/mlp_solver.prototxt -weights=./trainedModels/fcnModel.caffemodel

