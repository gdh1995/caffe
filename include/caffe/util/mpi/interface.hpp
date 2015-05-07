#ifndef CAFFE_UTIL_MPI_INTERFACE_HPP_
#define CAFFE_UTIL_MPI_INTERFACE_HPP_

#include "caffe/common.hpp"
#include "caffe/blob.hpp"

namespace caffe {
namespace mpi {

// DType: DataType; CType: ContainerType
template <typename DType, typename Ctype>
class BaseWorker {
 public:
  typedef const CType<Dtype> &CDataRef;
  virtual void sync  (CDataRef data);
  virtual void signal(CDataRef data);

 private:
  DISABLE_COPY_AND_ASSIGN(Worker);
}

class Interface {
 public:
  enum WorkerType { SELF_ONLY, PARENT, CHILD };
  typedef void (Handler)(void *data);
  struct HandlerWrapper {
    Handler *func;
    void *data;
  };
  template <typename Dtype>
  typedef vector<shared_ptr<Blob<Dtype> > > SharedParams;
  template <typename Dtype>
  typedef const SharedParams<DType> &SharedParamsRef;

  static inline SetDeviceList(const int *id_list, const int count) {
    mpi.device_list_ = id_list;
    mpi.device_count_ = count;
  }
  static inline GetDevice(const int index) {
    return mpi.device_list_ != NULL ? mpi.device_list_[index] : -1;
  }

  static void setup_handler(WorkerType type, Handler *func, void *data);
  static void trigger();

  template <typename Dtype>
  static WorkerType fork(const SolverParameter& param,
      SharedParamsRef<Dtype> net_params) {
    int data_copy = param.data_parallel(), model_copy = param.model_parallel();
    CHECK_GE(data_copy, 1) << "Copy number is invalid.";
    CHECK_GE(model_copy, 1) << "Copy number is invalid.";
    mpi.data_partition_ = data_copy;
    mpi.model_partition_ = model_copy;
    if (!mpi.check_for_fork()) {
      return mpi.worker_type_ = SELF_ONLY;
    };
    mpi.worker_ = mpi.do_fork();
    mpi.trigger();
    return mpi.worker_type_;
  }

  template <typename Dtype>
  static void sync(SharedParamsRef<Dtype> net_params) {
    if (mpi.worker_type_ != SELF_ONLY) {
      static_cast<BaseWorker<DType, vector<shared_ptr<Blob> > > >(mpi.worker_)
          .sync(net_params);
    }
  }

  template <typename Dtype>
  static void signal(SharedParamsRef<Dtype> net_params) {
    if (mpi.worker_type_ != SELF_ONLY) {
      static_cast<BaseWorker<DType, vector<shared_ptr<Blob> > > >(mpi.worker_)
          .signal(net_params);
    }
  }

  static inline WorkerType worker_type() { return mpi.worker_type_; }
  static inline int data_partition() { return mpi.data_partition_; }
  static inline int model_partition() { return mpi.model_partition_; }
  static inline int device_count() { return mpi.device_count_; }
  static inline int device_list() { return mpi.device_list_; }

 private:
  bool check_for_fork();
  template <typename Dtype>
  void *do_fork(SharedParamsRef<Dtype> net_params) const;

  void trigger();

  mutable WorkerType worker_type_;
  int data_partition_;
  int model_partition_;

  int device_count_;
  const int *device_list_;

  vector<HandlerWrapper> handlers_[3];

  void *worker_;

  static Interface mpi;
  Interface();
  DISABLE_COPY_AND_ASSIGN(Interface);
};

}  // namespace mpi

typedef mpi::Interface MPI;

}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_INTERFACE_HPP_
