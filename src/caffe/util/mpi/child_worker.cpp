#include <vector>
#include <errno.h>
#include <signal.h>

#include "caffe/util/mpi/worker.hpp"
#include "caffe/blob.hpp"

static void at_child_exit();

namespace caffe {
namespace mpi {

template <typename Dtype>
ChildWorker<Dtype>::ChildWorker(int child_index, int parent_pid,
    int data_size, char *parent_memory)
  : Worker<Dtype>(), child_index_(child_index), parent_pid_(parent_pid)
  , data_size_(data_size), parent_memory_(parent_memory)
  , memory_(parent_memory + data_size * child_index)
{
  block_signal_for_sync();
  ::signal(SIGTERM, exit);
  ::atexit(at_child_exit);
  
  LOG(INFO) << "Fork a child #" << child_index << ", map: " << (int*)memory_
      << ", parent: " << (int*)parent_memory_;
  LOG(INFO) << "    MPI: signal SYNC is " << SIGSYNC;
  if (Caffe::mode() == Caffe::GPU) {
    const int device_id = MPI::GetDevice(child_index);
    Caffe::SetDevice(device_id);
    set_peer_device(get_parent_device_id());
    LOG(INFO) << "Child #" << child_index << " uses the device #" << device_id;
  }
}

template <typename Dtype>
void ChildWorker<Dtype>::sync(CDataRef data) {
  BufferUnit *buffer;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    buffer = ((WorkerData *)memory_)->data;
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(buffer, data[i]->cpu_diff(), sizeof(Dtype) * count);
      buffer = buffer->next(count);
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    buffer = ((WorkerData *)((Dtype **)parent_memory_)[0])
        ->next(data_size_ * child_index_)->data;
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(buffer, data[i]->gpu_diff(), sizeof(Dtype) * count,
          cudaMemcpyDeviceToDevice));
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
  volatile const BufferUnit *parent_cpu_buffer;
  const BufferUnit *parent_gpu_buffer;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    parent_cpu_buffer = ((volatile const WorkerData *)parent_memory_)->data;
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(data[i]->mutable_cpu_data(), (Dtype *)parent_cpu_buffer->data,
          sizeof(Dtype) * count);
      parent_cpu_buffer = parent_cpu_buffer->nextv(count);
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    parent_gpu_buffer = ((WorkerData *)((Dtype **)parent_memory_)[0])->data;
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(data[i]->mutable_gpu_data(), parent_gpu_buffer,
          sizeof(Dtype) * count, cudaMemcpyDeviceToDevice));
      parent_gpu_buffer = parent_gpu_buffer->next(count);
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
