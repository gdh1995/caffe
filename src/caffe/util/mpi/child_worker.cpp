#include <vector>
#include <errno.h>
#include <signal.h>

#include "caffe/util/mpi/worker.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

static void at_child_exit();

namespace caffe {
namespace mpi {

template <typename Dtype>
ChildWorker<Dtype>::ChildWorker(int child_index, int parent_pid,
    int data_size, char *memory, const char *parent_memory)
  : Worker<Dtype>(), child_index_(child_index), parent_pid_(parent_pid)
  , data_size_(data_size), memory_(memory), parent_memory_(parent_memory)
{
  block_signal_for_sync();
  ::signal(SIGTERM, exit);
  ::atexit(at_child_exit);
  
  LOG(INFO) << "Fork a child #" << child_index << ", map: " << (int*)memory;
  LOG(INFO) << "    MPI: signal SYNC is " << SIGSYNC;
  if (Caffe::mode() == Caffe::GPU) {
    const int device_id = MPI::GetDevice(child_index);
    Caffe::SetDevice(device_id);
    LOG(INFO) << "Child #" << child_index << " uses the device #" << device_id;
  }
}

template <typename Dtype>
void ChildWorker<Dtype>::sync(CDataRef data) {
  WorkerData *worker = (WorkerData *)memory_;
  BufferUnit *buffer = worker->data;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(buffer, data[i]->cpu_diff(), sizeof(Dtype) * count);
      buffer = buffer->next(count);
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(buffer, data[i]->gpu_diff(),
          sizeof(Dtype) * count, cudaMemcpyDeviceToHost));
      buffer = buffer->next(count);
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }

  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);

  union sigval rc_val;
  rc_val.sival_int = 1;
  sigqueue(parent_pid_, SIGSYNC, rc_val);
  for (int sig; ; ) {
    if (0 != sigwait(&wait_set, &sig)) {
    } else if (sig == SIGSYNC) {
      break;
    }
  }
  work(data);
}

template <typename Dtype>
void ChildWorker<Dtype>::work(CDataRef data) {
  volatile const WorkerData *const parent_worker =
      (volatile const WorkerData *)parent_memory_;
  volatile const BufferUnit *parent_buffer = parent_worker->data;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      Dtype *const data_ptr = data[i]->mutable_cpu_data();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(data_ptr, (Dtype *)parent_buffer->data, sizeof(Dtype) * count);
      parent_buffer = parent_buffer->nextv(count);
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(data[i]->mutable_gpu_data(),
          const_cast<const BufferUnit *>(parent_buffer),
          sizeof(Dtype) * count, cudaMemcpyHostToDevice));
      parent_buffer = parent_buffer->nextv(count);
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void ChildWorker<Dtype>::setInterface(Interface &interface) {
  interface.setWorkerType(Interface::CHILD);
  interface.setChildIndex(child_index_);
}

INSTANTIATE_CLASS(ChildWorker);

}  // namespace mpi
}  // namespace caffe

using namespace caffe::mpi;

void at_child_exit() {
  LOG(INFO) << "Child #" << caffe::MPI::child_index() << " exit.";
}
