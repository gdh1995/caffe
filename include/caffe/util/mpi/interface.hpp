#ifndef CAFFE_UTIL_MPI_INTERFACE_HPP_
#define CAFFE_UTIL_MPI_INTERFACE_HPP_

#include "caffe/common.hpp"
#include "caffe/blob.hpp"

namespace caffe {
namespace mpi {

class Interface;

class SafeClass {
  virtual ~SafeClass() {}
}

template <typename Dtype>
class BaseWorker : public SafeClass {
 public:
  inline BaseWorker() {}
  typedef const vector<shared_ptr<Blob<Dtype> > > &CDataRef; 
  virtual void sync  (CDataRef data) = 0;
  virtual void signal(CDataRef data) = 0;
  virtual void setInterface(Interface &interface) = 0;

 private:
  DISABLE_COPY_AND_ASSIGN(BaseWorker);
};

class Interface {
 public:
  Interface();
  ~Interface();

  enum WorkerType { SELF_ONLY, PARENT, CHILD };
  typedef void (Handler)(void *data);
  struct HandlerWrapper {
    Handler *func;
    void *data;
  };

  static inline void SetDeviceList(const int *id_list, const int count) {
    mpi.device_list_ = id_list;
    mpi.device_count_ = count;
  }
  static inline int GetDevice(const int index) {
    return mpi.device_list_[index];
  }

  static void setup_handler(WorkerType type, Handler *func, void *data);

  template <typename Dtype>
  static WorkerType fork(const SolverParameter& param,
      const vector<shared_ptr<Blob<Dtype> > > &net_params) {
    int data_copy = param.data_parallel(), model_copy = param.model_parallel();
    CHECK_GE(data_copy, 1) << "Data parallel number is invalid.";
    CHECK_GE(model_copy, 1) << "Model parallel number is invalid.";
    mpi.data_partition_ = data_copy;
    mpi.model_partition_ = model_copy;
    if (!mpi.check_for_fork()) {
      mpi.worker_ = new SelfWorker<Dtype>();
      return mpi.worker_type_ = SELF_ONLY;
    };
    mpi.worker_ = mpi.do_fork(net_params);
    static_cast<BaseWorker<Dtype>*>(mpi.worker_)->setInterface(mpi);
    mpi.triggerHandlers();
    return mpi.worker_type_;
  }

  template <typename Dtype>
  static void sync(const vector<shared_ptr<Blob<Dtype> > > &net_params) {
    if (mpi.worker_type_ != SELF_ONLY) {
      static_cast<BaseWorker<Dtype>*>(mpi.worker_)->sync(net_params);
    }
  }

  template <typename Dtype>
  static void signal(const vector<shared_ptr<Blob<Dtype> > > &net_params) {
    if (mpi.worker_type_ != SELF_ONLY) {
      static_cast<BaseWorker<Dtype>*>(mpi.worker_)->signal(net_params);
    }
  }

  static inline WorkerType worker_type() { return mpi.worker_type_; }
  static inline int data_partition() { return mpi.data_partition_; }
  static inline int model_partition() { return mpi.model_partition_; }
  static inline int device_count() { return mpi.device_count_; }
  static inline const int *device_list() { return mpi.device_list_; }
  static int child_index() { return mpi.child_index_; }

  void setWorkerType(WorkerType t) { worker_type_ = t; }
  void setChildIndex(int index) { child_index_ = index; }

 private:

  bool check_for_fork();
  template <typename Dtype>
  SafeClass *do_fork(const vector<shared_ptr<Blob<Dtype> > > &net_params) const;

  void triggerHandlers();

  WorkerType worker_type_;
  int child_index_;
  int data_partition_;
  int model_partition_;

  int device_count_;
  const int *device_list_;

  vector<HandlerWrapper> handlers_[3];

  SafeClass *worker_;

  static Interface mpi;

  DISABLE_COPY_AND_ASSIGN(Interface);
};

}  // namespace mpi

typedef mpi::Interface MPI;

}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_INTERFACE_HPP_
