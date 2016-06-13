#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;

#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

//#include "caffe/data_layers.hpp"


#include <boost/filesystem.hpp>

using namespace cv;



vector<Mat> composePredictions(const vector<Mat>& patches, const vector<Mat>&images, int dataWidth, int dataHeight){

    //in this function we get a set of patch predictions and
    //compose them into a set of larger images

    vector<Mat> output;

    //find size of every individual patch
    int patchW = patches[0].cols;
    int patchH = patches[0].rows;

    //start by the first patch in the list
    size_t patchPointer=0;

    //for all the required output images...
    for (int n=0;n<images.size();n++){


        int imgW = images[n].cols;
        int imgH = images[n].rows;

        //define how many rows and cols of patches fit in the image
        int patchCols = floor((imgW - (dataWidth-patchW))/(float)patchW);
        int patchRows = floor((imgH - (dataHeight-patchH))/(float)patchH);


	//create an output map for the entire image
        Mat fullMap = Mat::zeros(patchRows*patchH,patchCols*patchW,patches[0].type());

	//copy the patches into the corresponding location in the larger image
        for (int i=0;i<patchRows && patchPointer<patches.size();i++)
            for (int j=0;j<patchCols && patchPointer<patches.size();j++){

                Mat submap = fullMap.rowRange(i*patchH,(i+1)*patchH).colRange(j*patchW,(j+1)*patchW);

                patches[patchPointer].copyTo(submap);

                patchPointer++;

            }

	//add to the list of output images
        output.push_back(fullMap);


    }

    return output;

}




vector<string> getFilesInDirectory(string dir){

    //Returns a sorted list of all the files in a given directory

    vector<string> out;

    boost::filesystem::path root(dir);

    CHECK(boost::filesystem::exists(root)) << dir << " does not exist.";
    CHECK(boost::filesystem::is_directory(root)) << root.c_str() << " is not a directory.";

    boost::filesystem::directory_iterator it(root);
    boost::filesystem::directory_iterator endit;

    while(it != endit) {
            if(boost::filesystem::is_regular_file(*it)
                    // && it->path().extension() == ext)
                    )
                out.push_back(it->path().string());
            ++it;
    }

    //just in case let's order the entries
    sort(out.begin(),out.end());

    return out;
}




vector<Mat> readFromFolder(string dir){
    //read all the images in a folder

    vector<Mat> outputImgs;

    //obtain the names
    vector<string> names = getFilesInDirectory(dir);
    
    //read the files using OpenCV
    for (size_t i=0;i<names.size();i++){

        Mat img = imread(names[i]);
        outputImgs.push_back(img);
    }

    return outputImgs;
}



int main(int argc, char** argv)
{

    //do patchwise predictions of a series of images
    //first argument: network definition location
    //second argument: network weights snapshot location
    //third argument: prefix to save results




    //location of the network definition in a prototxt file
    string netLocation;
    //location of the saved weight model to load
    string weightsLocation;
    //prefix to save results
    string savePrefix;

    if (argc==4){
        netLocation=argv[1];
        weightsLocation=argv[2];
	savePrefix=argv[3];
    } else {
        exit(1);

    }

    //use GPU
    Caffe::set_mode(Caffe::GPU);

    // Load network
    Net<float> net(netLocation,caffe::TEST);

    // Load pre-trained network
    net.CopyTrainedLayersFrom(weightsLocation);


    //patch height and width
    int patchH;
    int patchW;

    //do we have reference data?
    bool withLabels=net.has_blob("label");

    //get data blob information
    const boost::shared_ptr<Blob<float> >& dataBlob = net.blob_by_name("data");
    int dataHeight =dataBlob->height();
    int dataWidth =dataBlob->width();

    //read batch size from layer shape
    int batchSize = dataBlob->num();

    //obtain image directory from network specification
    const boost::shared_ptr<Layer<float> >& dataLayer=net.layer_by_name("data");
    string image_dir = dataLayer->layer_param().image_pair_data_param().image_dir();
   
    //read the input images from text file
    vector<Mat> images = readFromFolder(image_dir);
    
    //Get output probability blob information
    const boost::shared_ptr<Blob<float> >& probBlob = net.blob_by_name("prob");


    //if we assume square output, we can deduce the dimensions
    //from the output blob
    //patchH = patchW = sqrt(probBlob->channels()/labelSize);
    patchH = patchW = sqrt(probBlob->num()*probBlob->channels()*probBlob->height()*probBlob->width()/batchSize);


    //initialize vectors with predicted patches and corresponding labels
    vector<Mat> predictedPatches;
    vector<Mat> labelPatches;

    //compute the total number of patches required
    int totalNPatches=0;
    for (int i=0;i<images.size();i++){

        //dimensions of image from which patches are extracted
        int imgW = images[i].cols;
        int imgH = images[i].rows;

        //define how many rows and cols of patches fit in the image
        int patchCols = floor((imgW - (dataWidth-patchW))/(float)patchW);
        int patchRows = floor((imgH - (dataHeight-patchH))/(float)patchH);

        totalNPatches += patchCols*patchRows;
    }

    
    //compute number of iterations required to predict the whole dataset
    int max_iter = ceil(totalNPatches/(float)batchSize);
    LOG(INFO)<<"Number of iterations: "<<max_iter;


    //Iterations to predict all required patches:
    ////////////////////////////

    //start computing time
    double t = (double)getTickCount();


    //for all required iterations...
    for (int iterCount=0;iterCount<max_iter;iterCount++){

        //do neural network prediction
	float loss = 0.0;
	vector<Blob<float>*> results = net.ForwardPrefilled(&loss);

        //get output probabilities
        const float* probs_data = probBlob->cpu_data();

        //number of elements in the batch
        int number = dataBlob->num();

        //number of output variables in every element of the batch
        int outputsPerBatch = probBlob->num()*probBlob->channels()*probBlob->height()*probBlob->width()/batchSize;

        //number of labels
        int n_labels = outputsPerBatch;

        //get labels if appropriate
        const float* label_data=0;
        if (withLabels){
            const boost::shared_ptr<Blob<float> >& labelLayer = net.blob_by_name("label");
            label_data = labelLayer->cpu_data();
        }

        //for every element in the batch
        for (int n=0;n<number;n++){

            Mat_<float> patch(patchH,patchW);

            //contruct classification map
            for (int i=0;i<n_labels;i++){

                    patch(i/patchW,i%patchW)=probs_data[n*outputsPerBatch+i];
            }

            //add to list of classified patches
            predictedPatches.push_back(patch);

            if (withLabels){
                Mat_<uchar> labels(patchH,patchW);

                for (int i=0;i<n_labels;i++){
                    labels(i/patchW,i%patchW)=label_data[n*outputsPerBatch+i]*255.0;
                }
                labelPatches.push_back(labels);
            }
        }
		
    } //iterations to predict individual patches


    //show time
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl;
    /////////////////////////////////////////////



    //compose predictions into appropriate image sizes
    vector<Mat> outputMaps = composePredictions(predictedPatches,images,dataWidth,dataHeight);

    //compose reference maps if available
    vector<Mat> referenceMaps;
    if (withLabels)
        referenceMaps = composePredictions(labelPatches,images,dataWidth,dataHeight);


    /////////////////////////////////////////////




    //save output

    for (size_t i =0; i<outputMaps.size(); i++){
        stringstream s; s<<i;

	//save soft classification map scaled to [0,256)
        imwrite(savePrefix+"softClassif"+s.str()+".png",outputMaps[i]*255);

        //save probabilities to a data file

        vector<double> probsToFile;
        for (int y=0;y<outputMaps[i].rows;y++){
            for (int x=0;x<outputMaps[i].cols;x++){

                double prob = outputMaps[i].at<float>(y,x);
                probsToFile.push_back(prob);
                probsToFile.push_back(1.0-prob);

            }
        }

        ofstream data_file;
        string dataFileName = savePrefix+"prob_data"+s.str()+".dat";
        data_file.open(dataFileName.c_str(), ios_base::out | ios_base::binary);
        data_file.write(reinterpret_cast<char*>(&probsToFile[0]),probsToFile.size()*sizeof(double));
        data_file.close();


	//save binary classification map (0/255)

        Mat binary;
        threshold(outputMaps[i],binary,1.0/2.0,255.0,THRESH_BINARY);
        imwrite(savePrefix+"classif"+s.str()+".png",binary);


        if (withLabels){

	    //save the binary labels 
            Mat label_binary;
            threshold(referenceMaps[i],label_binary,0,255,THRESH_BINARY);
            imwrite(savePrefix+"reference"+s.str()+".png",label_binary);
            
        }

    }

    return 0;
}
