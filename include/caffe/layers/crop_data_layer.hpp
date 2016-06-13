#ifndef CAFFE_CROP_DATA_LAYER_HPP_
#define CAFFE_CROP_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {


//this layer spatially crops the output of a data layer.
//the size of the crop can be specified, or how much to crop
//(with negative values in the crop size parameter)


template <typename Dtype>
class CropDataLayer : public Layer<Dtype> {
 public:
  explicit CropDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CropData"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}


  int crop_h_, crop_w_;
};

}  // namespace caffe

#endif  // CAFFE_CROP_DATA_LAYER_HPP_
