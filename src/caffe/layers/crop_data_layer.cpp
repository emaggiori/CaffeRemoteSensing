#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/crop_data_layer.hpp"


#include <iostream>
using namespace std;

namespace caffe {

template <typename Dtype>
void CropDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


	//obtain height and width from layer specification
	int param_h = this->layer_param_.crop_data_param().h();
	int param_w = this->layer_param_.crop_data_param().w();

	// if they are both positive we assume it's the size of the cropped area
	if (param_h > 0 && param_w>0) {

		crop_h_=param_h;
		crop_w_=param_w;
	} else {
	//otherwise we assume it's how much we must subtract at every
	//side of the crop

		int bottom_h = bottom[0]->height();
		int bottom_w = bottom[0]->width();

		crop_h_=bottom_h + param_h;
		crop_w_=bottom_w + param_w;
	}
}

template <typename Dtype>
void CropDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


    //if there's a second blob we use it's shape to crop
    if (bottom.size()==2){
        top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(),
           bottom[1]->width());
    } else {
	//otherwise we use the specified parameters
        top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), crop_h_, crop_w_);
    }
}




template <typename Dtype>
void CropDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


  //pointers to bottom data to read from 
  //and top data to write to
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  //compute top left corner of the crop
  int h_start = bottom[0]->height()/2.0 - floor(crop_h_/2.0);
  int w_start = bottom[0]->width()/2.0 - floor(crop_w_/2.0);

  //copy data to top blob...

  int K = top[0]->channels();
  int W = top[0]->width();
  int H = top[0]->height();

  for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < K; ++c) {
          for (int h = 0; h < H; ++h) {

              for (int w =0 ; w< W;w++){
                  *(top_data + top[0]->offset(n, c, h, w)) = *(bottom_data + bottom[0]->offset(n, c, h_start + h, w_start + w));

              }

          }
      }

  }


}





INSTANTIATE_CLASS(CropDataLayer);
REGISTER_LAYER_CLASS(CropData);

}  // namespace caffe
