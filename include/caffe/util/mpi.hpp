#ifndef CAFFE_UTIL_MPI_HPP_
#define CAFFE_UTIL_MPI_HPP_

#include "caffe/common.hpp"

namespace caffe {

class MPInterface {
 public:
  enum ForkStatus { NONE, CHILD, PARENT };
  typedef void (*Callback)(void *data);

  static inline ForkStatus fork(const int data_copy, const int model_copy) {
    set_copy(data_copy, model_copy);
    child_index_ = 0;
    return fork_stat_ = do_fork();
  }

  template <typename Dtype>
  static void sync(const vector<shared_ptr<Blob<Dtype> > > &data);
  template <typename Dtype>
  static void signal(const vector<shared_ptr<Blob<Dtype> > > &data);

  static inline ForkStatus fork_stat() { return return fork_stat_; }
  static inline int child_index() { return child_index_; }
  static inline int data_partition() { return data_partition_; }
  static inline int model_partition_() { return model_partition_; }
  static void setup_onfork(Callback func, void *data);

 private:
  static ForkStatus do_fork();
  static void set_copy(const int data_copy, const int model_copy);

  static ForkStatus fork_stat_;
  static int child_index_;
  static int data_partition_;
  static int model_partition_;

  DISABLE_COPY_AND_ASSIGN(MPInterface);  
};

typedef MPInterface MPI;

}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_HPP_
