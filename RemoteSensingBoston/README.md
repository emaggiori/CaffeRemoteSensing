# Caffe for remote sensing

## Boston dataset

These networks were trained on the [Massachusetts buildings dataset](https://www.cs.toronto.edu/~vmnih/data/).

To execute the CNN predictions you must first point to the right location of the dataset in the network configuration file (e.g., fullyconv.prototxt).

Then run the tool "predictBinary" (e.g., build/tools/predictBinary RemoteSensingBoston/fullyconv/fullyconv.protoxt RemoteSensingBoston/fullyconv/savedmodel.caffemodel RemoteSensingBoston/results/).
