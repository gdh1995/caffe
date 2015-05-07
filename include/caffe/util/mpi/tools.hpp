#ifndef CAFFE_UTIL_MPI_TOOLS_HPP_
#define CAFFE_UTIL_MPI_TOOLS_HPP_

#include "caffe/common.hpp"
#include "caffe/blob.hpp"

namespace caffe {
namespace mpi {

template <typename Dtype>
void calc_shared_mem(const vector<shared_ptr<Blob<Dtype> > > &net_params) {
  if (data_partition_ <= 1) {
    return;
  }
  int sum = 0;
  for (int i = 0; i < net_params.size(); i++) {
    Blob<Dtype> *blob = net_params[i].get();
    int len = sizeof(Dtype) * blob->count();
    len = (len + 15) / 16;
    sum += sizeof(int) * 4 + len * 16;
  }
  child_mem_size = sizeof(int) * 4 + sum;
  shared_mem_size = child_mem_size * (data_partition_ + 1);
  LOG(INFO) << "ChildUnit: " << child_mem_size << " @ " << net_params.size();
}



}  // namespace mpi
}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_TOOLS_HPP_
