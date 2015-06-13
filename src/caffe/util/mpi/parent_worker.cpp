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
  , children_(children), memory_(memory)
{ 
  set_for_clean(sizeof(Dtype), this);
  block_signal_for_sync();
  ::atexit(clean_at_exit);
  ::signal(SIGHUP, exit);
  ::signal(SIGINT, exit);
  ::signal(SIGTERM, exit);
  ::signal(SIGQUIT, exit);

  LOG(INFO) << "Parent holds on shared memory " << children_size * data_size
      << " Bytes";
  LOG(INFO) << "    MPI: signal SYNC is " << SIGSYNC;
  Caffe::set_mode(Caffe::CPU);
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
  WorkerData *worker = ((WorkerData *)memory_);
  BufferUnit *buffer = worker->data;
  const WorkerData *child_worker = worker;
  const BufferUnit *child_buffer = child_worker->data;
  const int count = (data_size_ - WorkerData::BufferDataOffset) / sizeof(Dtype);
  for (int i = 1; i < children_size_; i++) {
    child_worker = child_worker->next(data_size_);
    child_buffer = child_worker->data;
    caffe_axpy(count, (Dtype)1, (const Dtype *)child_buffer, (Dtype *)buffer);
  }
  caffe_scal(count, (Dtype)1 / children_size_, (Dtype *)buffer);

  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    Dtype *diff_ptr = data[i]->mutable_cpu_diff();
    caffe_copy(count, (const Dtype *)buffer, diff_ptr);
    buffer = buffer->next(count);
  }
}

template <typename Dtype>
void ParentWorker<Dtype>::signal(CDataRef data) {
  WorkerData *worker = (WorkerData *)memory_;
  BufferUnit *buffer = worker->data;
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    const Dtype *cpu_data = data[i]->cpu_data();
    caffe_copy(count, cpu_data, (Dtype *)buffer);
    buffer = buffer->next(count);
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
  interface.setHostMemory(memory_, children_size_ * data_size_);
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
