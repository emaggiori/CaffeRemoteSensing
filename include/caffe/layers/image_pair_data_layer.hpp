#ifndef IMAGEPAIR_DATA_LAYER_HPP_
#define IMAGEPAIR_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/features2d/features2d.hpp>

namespace caffe {


//This layer groups pair of patches (image/labels) extracted from an image dataset.
//If the layer is configured to output a single blob, labels are ignored.

//The dataset is specified with directories that contain the images.

//The size of the labeled patch might be smaller than the image patch. In that case,
//the label patch outputted is centered in the image patch.

//If "random" is activated, patches are sampled at random locations from all the dataset.

//If classification is binary, nonzero entries are considered to be "class 1"
//and zero entries are "class 0)
//If classification is multi-class, the label patch at every location is the class index (0,1,2,3).
//In the input labeled image the classes can be encoded this way directly or with a step (e.g, 0,10,20)
//to make the visualization easier (which should be indicated in the "class_step") variable
 


template <typename Dtype>
class ImagePairDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImagePairDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImagePairDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImagePairData"; }

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:

  //function where the batch is loaded with input data
  virtual void load_batch(Batch<Dtype>* batch);

  //auxiliary function to read files from directory
  vector<string> getFilesInDirectory(string dir);

  //list with the locations of image/label pairs
  vector<std::pair<std::string, std::string> > lines_;

  //the images themselves
  vector<std::pair<cv::Mat, cv::Mat> > images;


  //patch size parameters
  int h_win1;
  int w_win1;
  int h_win2;
  int w_win2;

  int channels;

  //do we sample randomly?
  bool random;
  //random number generator
  cv::RNG rng;

  //is this a multiclass or a binary problem?
  bool is_multiclass;
  //class step to encode multiclass labels
  int class_step;

  //image transformation parameters
  //(to subtract the mean and scale)
  float scaleFactor;
  float meanValue;


  //index of the current image and positions in the image
  //(when reading is sequential and not random)
  int lines_id_;
  int currentLeftX1;
  int currentTopY1;
  
};


}  // namespace caffe

#endif
