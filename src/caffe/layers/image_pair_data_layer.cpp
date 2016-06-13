#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/image_pair_data_layer.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"




#include <boost/filesystem.hpp>



namespace caffe {
	
	
	
using namespace cv;
using namespace std;
	



template <typename Dtype>
vector<string> ImagePairDataLayer<Dtype>::getFilesInDirectory(string dir){

    //Returns a sorted list of all the files in a given directory

    vector<string> out;

    boost::filesystem::path root(dir);

    CHECK(boost::filesystem::exists(root)) << dir << " does not exist.";
    CHECK(boost::filesystem::is_directory(root)) << root.c_str() << " is not a directory.";

    boost::filesystem::directory_iterator it(root);
    boost::filesystem::directory_iterator endit;

    while(it != endit) {
            if(boost::filesystem::is_regular_file(*it))
                out.push_back(it->path().string());
            ++it;
    }

    //just in case let's order the entries
    sort(out.begin(),out.end());

    return out;
}

	
	

template <typename Dtype>
ImagePairDataLayer<Dtype>::~ImagePairDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImagePairDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


    //obtain image and label directories

    const string& image_dir = this->layer_param_.image_pair_data_param().image_dir();
    //this is compulsory
    CHECK_GT(image_dir.size(), 0) << "There must be an image directory";

    const string& label_dir = this->layer_param_.image_pair_data_param().label_dir();
    //this might be required
    if (this->output_labels_){
        CHECK_GT(label_dir.size(), 0) << "There must be a label directory";
    }


    //obtain file names

    vector<string> filenames1 = getFilesInDirectory(image_dir);

    vector<string> filenames2;
    if (this->output_labels_){
        filenames2 = getFilesInDirectory(label_dir);
        CHECK_EQ(filenames1.size(),filenames2.size()) << 
        "There must be an equal number of data and reference images.";
    }



    //create lists with the color image file names 
    //paired with the corresponding labels (if applicable)
    for (size_t i=0;i<filenames1.size();i++) {

        if (this->output_labels_){
            lines_.push_back(std::make_pair(filenames1[i], filenames2[i]));
        } else {
            lines_.push_back(std::make_pair(filenames1[i], ""));
        }

    }




  //open images using OpenCV. Could be improved by using GDAL

  for (size_t i=0;i<lines_.size();i++) {


    string filename1 = lines_[i].first;
    string filename2 = lines_[i].second;

    Mat img1 = imread(filename1);
    CHECK(img1.data) << "Could not read image "<<filename1;

    Mat img2;
    if (this->output_labels_){ //if output labels
        img2 = imread(filename2,CV_LOAD_IMAGE_GRAYSCALE);
        CHECK(img2.data) << "Could not read image "<<filename2;
    }

    images.push_back(std::make_pair(img1, img2));
  }
  
  //print the number of images
  LOG(INFO) << "A total of " << lines_.size() << " images.";



  //obtain size parameters

  h_win1=this->layer_param_.image_pair_data_param().h_img();
  w_win1=this->layer_param_.image_pair_data_param().w_img();
  h_win2=this->layer_param_.image_pair_data_param().h_map();
  w_win2=this->layer_param_.image_pair_data_param().w_map();

  is_multiclass = this->layer_param_.image_pair_data_param().multiclass();
  class_step = this->layer_param_.image_pair_data_param().class_step();


  //if the dimensions of the label window are unspecified,
  //just take the same dimensions as for the data
  if (h_win2==0 && w_win2==0){
      h_win2=h_win1;
      w_win2=w_win1;
  }


  //some checks about the sizes
  CHECK(h_win1%2==h_win2%2)<<
     "The height parity of predicted and input patch must match";
  CHECK(w_win1%2==w_win2%2)<<
    "The width parity of predicted and input patch must match";
  CHECK(h_win1>=h_win2)<<
    "The dimension of input patch must be equal or greater than predicted patch";
  CHECK(w_win1>=w_win2)<<
    "The dimension of input patch must be equal or greater than predicted patch";

  //we only accept color images at the moment
  channels=this->layer_param_.image_pair_data_param().channels();
  CHECK(channels==3)<<"Only one/three channel input accepted";

  //transformation parameters
  scaleFactor = this->layer_param_.image_pair_data_param().scale();
  meanValue = this->layer_param_.image_pair_data_param().mean();

  //do we want to sample random patches or advance sequentially?
  this->random=this->layer_param_.image_pair_data_param().random();

  //we disable cropping option
  const int crop_size = this->layer_param_.transform_param().crop_size();
  CHECK(crop_size==0)<<"Crop size must be zero";

  //read batch size
  const int batch_size = this->layer_param_.image_pair_data_param().batch_size();
  
  //modify size of data blob
  top[0]->Reshape(batch_size, channels, h_win1, w_win1);

  //modify size of prefetch blobs
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
	this->prefetch_[i].data_.Reshape(
	   batch_size, channels, h_win1,w_win1);

  //do the same for labels if applicable
  if (this->output_labels_){
      //modify size of label blob and label prefetch blob
      top[1]->Reshape(batch_size, 1, h_win2, w_win2);
      for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	    this->prefetch_[i].label_.Reshape(batch_size, 1, h_win2, w_win2);
  }

  
  //the current line in the file
  lines_id_ = 0;

  //the current patch position (only makes sense
  //if we advance sequentially)
  currentLeftX1 = 0;
  currentTopY1 = 0;

}




// This function is used to called to prefetch the data
template <typename Dtype>
void ImagePairDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {


  CHECK(batch->data_.count());
  
  //get pointers to modify data/label blobs
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;
  if (this->output_labels_){
        top_label = batch->label_.mutable_cpu_data();
  }
  

  //get batch size
  ImagePairDataParameter image_pair_data_param = this->layer_param_.image_pair_data_param();
  const int batch_size = image_pair_data_param.batch_size();
  
  //number of files
  const int lines_size = lines_.size();
  
  //for every element required to fill the batch...
  for (int item_id = 0; item_id < batch_size; item_id++) {
	      
    //get an image index
    int whichImage;
    if (random){
        whichImage = rng.uniform(0,lines_size);
    } else {
        whichImage = lines_id_;
    }

    //point to that image from the list ("img1")
    Mat img1 = images[whichImage].first;

    //get dimensions of the image
    int w_img = img1.cols;
    int h_img = img1.rows;
  

    //extreme possible coordinates of left border of window1
    int minX = 0;
    int maxX = max(w_img - w_win1,0);
    //extreme possible coordinates of upper border of window1
    int minY = 0;
    int maxY = max(h_img - h_win1,0);


    //the indices of the patch to extract from img1
    int leftX1;
    int topY1;

    //a pointer to the patch
    Mat subimg;


    //a variable indicating if the patch is approved
    //to be used
    bool approvePatch=false;


    while(!approvePatch){

        //select a window
        if (this->random){
            //randomly		
            leftX1 = rng.uniform(minX,maxX+1);
            topY1 = rng.uniform(minY,maxY+1);
        } else {
            //in order
            leftX1=currentLeftX1;
            topY1=currentTopY1;
        }



        //select window
        subimg = img1.rowRange(topY1,topY1+h_win1).colRange(leftX1,leftX1+w_win1);

        //directly aprove patch
        approvePatch=true;
	
	//here we could add some code to conditionally accept the patch       

    }


    //fill the image data to the Caffe blob
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < h_win1; ++h) {
          for (int w = 0; w < w_win1; ++w) {
			
            //get the pixel value at this position and channel  
            Dtype pixel =  static_cast<Dtype>(subimg.at<cv::Vec3b>(h, w)[c]);

	    //the index where to put this value in the blob
            int index = ((item_id * channels + c) * h_win1 + h)* w_win1 + w;
			
            //set the value in the blob
            top_data[index] = (pixel-meanValue)*scaleFactor;          
          }
        }
    } //finished filling the image blob
    

    //only if the layer requires labels
    if (this->output_labels_){

   	//read the corresponding image in the pair 
	// (we keep "whichImage" from before)
        Mat img2 = images[whichImage].second;

	//recomputed the positions in the image, because
	//the label patch might be smaller and centered in the image patch
        int horizontalOffset = (w_win1 - w_win2)/2;
        int verticalOffset = (h_win1 - h_win2)/2;

        int leftX2 = leftX1 + horizontalOffset;
        int topY2 = topY1 + verticalOffset;

        //select window
        Mat subimg2 = img2.rowRange(topY2,topY2+h_win2).colRange(leftX2,leftX2+w_win2);

        for (int h = 0; h < h_win2; ++h) {
              for (int w = 0; w < w_win2; ++w) {

                //the index where to put this value in the blob
                int index = ((item_id * 1 + 0) * h_win2 + h)* w_win2 + w;

                if (!is_multiclass){
			//if binary problem coded with a single variable,
			//assign class 1 if reference is nonzero (foreground), 
			//and class 0 if reference is zero (background)
		        if (subimg2.at<uchar>(h, w)==0)
		            top_label[index] = 0;
		        else{
		            top_label[index] = 1;
		        }
		} else {
			//in the multiclass case we assign the class number 
			//divided by a step (for example, we could code classes as
			//0, 1, 2, 3 with step 1 or 0, 20, 40, 60 with step 20).
			//A large step might be useful to visualize the labeled images
			top_label[index] = subimg2.at<uchar>(h, w)/class_step;
		}

              }
        }

    } //finished filling the label blob



    //update patch indices in case it's not random
    if (!this->random){
        currentLeftX1+=w_win2;
        if (currentLeftX1 > maxX){
            currentLeftX1=0;
            currentTopY1+=h_win2;
            if (currentTopY1 > maxY){
                currentTopY1=0;
                lines_id_++;
                if (lines_id_ >= lines_size) {
                    lines_id_ = 0;
                }
            }
        }
    }


  } //finished filling one element in the prefetching batch


}

INSTANTIATE_CLASS(ImagePairDataLayer);
REGISTER_LAYER_CLASS(ImagePairData);

}  // namespace caffe
