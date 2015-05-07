#ifndef CAFFE_UTIL_MPI_WORKER_HPP_
#define CAFFE_UTIL_MPI_WORKER_HPP_

#include "caffe/util/mpi/interface.hpp"

namespace caffe {
namespace mpi {

template <typename DType, typename Ctype>
class Worker : public BaseWorker<DType, CType> {
 public:
  struct BufferUnit {
    DType data[0];

    BufferUnit *next(int count) { return (BufferUnit *)(data + count); }
    const BufferUnit *next(int count) const {
      return (const BufferUnit *)(data + count);
    }
  };

  struct WorkerData {
    enum WorkerStatus {WORKING, SYNCING};
    int status, pid;
    BufferUnit data[0];

    WorkerData *next(int byte_size) {
      return (WorkerData *)(((char *)this) + byte_size);
    }
    const WorkerData *next(int byte_size) const {
      return (const WorkerData *)(((const char *)this) + byte_size);
    }
  };

  static int GetParamsSize(CDataRef net_params);

  virtual void sync  (CDataRef data);
  virtual void signal(CDataRef data);

  static void InitBufferArray(BufferUnit *buffer, CDataRef data);

 private:
  DISABLE_COPY_AND_ASSIGN(Worker);
};

template <typename DType, typename Ctype>
class ParentWorker : public Worker<DType, CType> {
 public:
  ParentWorker(int children_size, const int *children, int data_size,
      char *memory);

  virtual void sync  (CDataRef data);
  virtual void signal(CDataRef data);

  void check_all_child();
  void clean();
  void work(CDataRef data);

 protected:
  const int children_size, data_size_;
  const int * const children_;
  char *const memory_;
  
 private:
  bool buffer_inited_;
  DISABLE_COPY_AND_ASSIGN(ParentWorker);
};

template <typename DType, typename Ctype>
class ChildWorker : public Worker<DType, CType> {
 public:
  ChildWorker(int child_index, int parent_pid, int data_size, char *memory,
      const char *parent_memory);

  virtual void sync  (CDataRef data);
  virtual void signal(CDataRef data);
  void work(CDataRef data);

 protected:
  const int child_index_, parent_pid_, data_size_;
  char *const memory_; 
  char *parent_memory_;

 private:
  bool buffer_inited_;
  DISABLE_COPY_AND_ASSIGN(ChildWorker);
};

}  // namespace mpi
}  // namespace caffe

#endif  // CAFFE_UTIL_MPI_WORKER_HPP_
