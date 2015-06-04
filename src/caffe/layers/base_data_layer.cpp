#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/mpi/interface.hpp"

template <typename Dtype>
static void on_fork(void *layer) {
  static_cast<caffe::BasePrefetchingDataLayer<Dtype> *>(layer)->init_skip();
}
template <typename Dtype>
static void on_not_fork(void *layer) {
  DLOG(INFO) << "Initializing prefetch";
  static_cast<caffe::BasePrefetchingDataLayer<Dtype> *>(layer)->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  this->skip_step_ = 0;
  MPI::setup_handler(MPI::CHILD, on_fork<Dtype>, this);
  MPI::setup_handler(MPI::SELF_ONLY, on_not_fork<Dtype>, this);
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread() {
  this->data_transformer_->InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  if (MPI::worker_type() == MPI::CHILD) {
    skip(skip_step_);
  }
  CreatePrefetchThread();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::init_skip() {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int skip_size_0 = MPI::child_index() * batch_size;
  this->skip_step_ = (MPI::data_partition() - 1) * batch_size;
  LOG(INFO) << "Layer " << this->layer_param_.name() << " skip " << skip_size_0;
  this->skip(skip_size_0);
  DLOG(INFO) << "Initializing prefetch after skip.";
  this->CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
