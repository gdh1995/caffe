#include <vector>
#include <errno.h>
#include <signal.h>

#include "caffe/util/mpi/worker.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

static void set_for_clean(int type_size, void *instance);
static void clean_at_exit();

namespace caffe {
namespace mpi {

template <typename Dtype>
ParentWorker<Dtype>::ParentWorker(int children_size, const int *children,
    int data_size, char *memory)
  : Worker<Dtype>(), children_size_(children_size), data_size_(data_size)
  , children_(children), memory_(memory), vec_x_(children_size)
  , first_params_(data_size), other_params_((children_size - 1) * data_size)
{ 
  set_for_clean(sizeof(Dtype), this);
  block_signal_for_sync();
  ::atexit(clean_at_exit);
  ::signal(SIGHUP, exit);
  ::signal(SIGINT, exit);
  ::signal(SIGTERM, exit);
  ::signal(SIGQUIT, exit);

  caffe_set(children_size_, ((Dtype)1.) / children_size_,
      (Dtype *)vec_x_.mutable_cpu_data());

  LOG(INFO) << "Parent holds on shared memory " << children_size * data_size
      << " Bytes";
  LOG(INFO) << "    MPI: signal SYNC is " << SIGSYNC;
  if (Caffe::mode() == Caffe::GPU) {
    const int device_id = MPI::GetDevice(0);
    Caffe::SetDevice(device_id);
    LOG(INFO) << "Parent uses the device #" << device_id;
    vec_x_.gpu_data();
  }
}

template <typename Dtype>
void ParentWorker<Dtype>::sync(CDataRef data) {
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  for (int sig, counter_sig_sync = 0; ; ) {
    if (0 != sigwait(&wait_set, &sig)) {
    } else if (sig == SIGSYNC) {
      ++counter_sig_sync;
    } else {
      continue;
    }
    if (counter_sig_sync >= children_size_) {
      break;
    }
  }
  work(data);
}

template <typename Dtype>
void ParentWorker<Dtype>::work(CDataRef data) {
  Dtype *vec_y;
  const Dtype *mat_A;
  const BufferUnit *buffer;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    vec_y = (Dtype *)first_params_.mutable_cpu_data();
    mat_A = (const Dtype *)other_params_.cpu_data();
    caffe_cpu_gemv<Dtype>(CblasNoTrans, children_size_ - 1,
        data_size_ / sizeof(Dtype), (Dtype)1., mat_A,
        (const Dtype *)vec_x_.cpu_data(), (Dtype)1. / children_size_, vec_y);

    buffer = ((WorkerData *)vec_y)->data;
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(data[i]->mutable_cpu_diff(), buffer, sizeof(Dtype) * count);
      buffer = buffer->next(count);
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    first_params_.set_cpu_data(memory_);
    other_params_.set_cpu_data(memory_ + data_size_);
    vec_y = (Dtype *)first_params_.mutable_gpu_data();
    mat_A = (const Dtype *)other_params_.gpu_data();
    caffe_gpu_gemv<Dtype>(CblasNoTrans, children_size_ - 1,
        data_size_ / sizeof(Dtype), (Dtype)1., mat_A,
        (const Dtype *)vec_x_.gpu_data(), (Dtype)1. / children_size_, vec_y);

    buffer = ((WorkerData *)vec_y)->data;
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(data[i]->mutable_gpu_diff(), buffer,
          sizeof(Dtype) * count, cudaMemcpyDefault));
      buffer = buffer->next(count);
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
void ParentWorker<Dtype>::signal(CDataRef data) {
  BufferUnit *buffer = ((WorkerData *)memory_)->data;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(buffer, data[i]->cpu_data(), sizeof(Dtype) * count);
      buffer = buffer->next(count);
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int i = 0; i < data.size(); i++) {
      const int count = data[i]->count();
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(buffer, data[i]->gpu_data(), sizeof(Dtype) * count,
          cudaMemcpyDeviceToHost));
      buffer = buffer->next(count);
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }

  union sigval rc_val;
  rc_val.sival_int = 2;
  for (int i = 0; i < children_size_; i++) {
    const pid_t pid = children_[i];
    sigqueue(pid, SIGSYNC, rc_val);
  }
}

template <typename Dtype>
void ParentWorker<Dtype>::clean() {
  union sigval rc_val;
  rc_val.sival_int = -2;
  for (int i = 0; i < children_size_; i++) {
    const pid_t pid = children_[i];
    sigqueue(pid, SIGTERM, rc_val);
  }
  LOG(INFO) << "Broadcast: exit";
}

template <typename Dtype>
void ParentWorker<Dtype>::setInterface(Interface &interface) {
  interface.setWorkerType(Interface::PARENT);
  interface.setChildIndex(0);
}

INSTANTIATE_CLASS(ParentWorker);

}  // namespace mpi
}  // namespace caffe

using namespace caffe::mpi;
static ParentWorker<float > *s_first_f = NULL;
static ParentWorker<double> *s_first_d = NULL;

static void set_for_clean(int type_size, void *instance) {
  if (type_size == 4) {
    s_first_f = (ParentWorker<float >*)instance;
  } else if (type_size == 8) {
    s_first_d = (ParentWorker<double>*)instance;
  }
}

void clean_at_exit() {
  if (s_first_f != NULL) {
    s_first_f->clean();
    s_first_f = NULL;
  }
  if (s_first_d != NULL) {
    s_first_d->clean();
    s_first_d = NULL;
  }
}
