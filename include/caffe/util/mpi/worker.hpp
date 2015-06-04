#ifndef CAFFE_UTIL_MPI_WORKER_HPP_
#define CAFFE_UTIL_MPI_WORKER_HPP_

#include "caffe/util/mpi/interface.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {
namespace mpi {

extern const int SIGSYNC;
const int SHARED_HOST_MEM_MIN_SIZE = 4096;

void block_signal_for_sync();
int get_parent_device_id();
int set_peer_device(int peer_id);

template <typename Dtype>
class Worker : public BaseWorker<Dtype> {
 public:
  typedef typename BaseWorker<Dtype>::CDataRef CDataRef;

  inline Worker(): BaseWorker<Dtype>() {}

  typedef struct BufferUnit {
    Dtype data[0];

    BufferUnit *next(int count) { return (BufferUnit *)(data + count); }
    const BufferUnit *next(int count) const {
      return (const BufferUnit *)(data + count);
    }
    volatile const BufferUnit *nextv(int count) const volatile {
      return (volatile const BufferUnit *)(data + count);
    }
  } BufferUnit;

  typedef struct WorkerData {
    BufferUnit data[0];
    static const int BufferDataOffset = 0;

    WorkerData *next(int byte_size) {
      return (WorkerData *)(((char *)this) + byte_size);
    }
    const WorkerData *next(int byte_size) const {
      return (const WorkerData *)(((const char *)this) + byte_size);
    }
  } WorkerData;

  static int GetParamsSize(CDataRef net_params);

  virtual void sync  (CDataRef data) { NOT_IMPLEMENTED; }
  virtual void signal(CDataRef data) { NOT_IMPLEMENTED; }

 private:
  DISABLE_COPY_AND_ASSIGN(Worker);
};

template <typename Dtype>
class SelfWorker : public Worker<Dtype> {
 public:
  typedef typename Worker<Dtype>::CDataRef CDataRef;
  typedef typename Worker<Dtype>::WorkerData WorkerData;
  typedef typename Worker<Dtype>::BufferUnit BufferUnit;

  SelfWorker();
  virtual void sync  (CDataRef data) {}
  virtual void signal(CDataRef data) {}
  virtual void setInterface(Interface &interface) {
    interface.setChildIndex(0);
    interface.setWorkerType(Interface::SELF_ONLY);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(SelfWorker);
};

template <typename Dtype>
class ParentWorker : public Worker<Dtype> {
 public:
  typedef typename Worker<Dtype>::CDataRef CDataRef;
  typedef typename Worker<Dtype>::WorkerData WorkerData;
  typedef typename Worker<Dtype>::BufferUnit BufferUnit;

  ParentWorker(int children_size, const int *children, int data_size,
      char *memory);

  virtual void sync  (CDataRef data);
  virtual void signal(CDataRef data);
  virtual void setInterface(Interface &interface);

  bool check_all_child();
  void clean();
  void work(CDataRef data);

 protected:
  const int children_size_, data_size_;
  const int * const children_;
  char *const memory_;
  SyncedMemory vec_x_, first_params_, other_params_;
  
 private:
  DISABLE_COPY_AND_ASSIGN(ParentWorker);
};

template <typename Dtype>
class ChildWorker : public Worker<Dtype> {
 public:
  typedef typename Worker<Dtype>::CDataRef CDataRef;
  typedef typename Worker<Dtype>::WorkerData WorkerData;
  typedef typename Worker<Dtype>::BufferUnit BufferUnit;

  ChildWorker(int child_index, int parent_pid, int data_size,
      const char *parent_memory);

  virtual void sync  (CDataRef data);
  virtual void signal(CDataRef data) {}
  virtual void setInterface(Interface &interface);

  void work(CDataRef data);

 protected:
  const int child_index_, parent_pid_, data_size_;
  char *const memory_; // used only when in CPU mode
  volatile const char *const parent_memory_;

 private:
  DISABLE_COPY_AND_ASSIGN(ChildWorker);
};

}  // namespace mpi
}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_WORKER_HPP_
