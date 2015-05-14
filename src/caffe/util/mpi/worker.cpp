#include <ctime>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <errno.h>
#include <sys/mman.h>
#include <signal.h>

#include "caffe/util/mpi/worker.hpp"
#include "caffe/blob.hpp"

static void set_for_clean(int type_size, void *instance);
static void clean_at_exit();
static void do_exit(int sig);
static void at_child_exit();


namespace caffe {
namespace mpi {

const int SIGSYNC = SIGRTMIN + 1;

template <typename Dtype>
int Worker<Dtype>::GetParamsSize(CDataRef net_params) {
  int sum = WorkerData::BufferDataOffset;
  for (int i = 0; i < net_params.size(); i++) {
    Blob<Dtype> *blob = net_params[i].get();
    int len = sizeof(Dtype) * blob->count();
    sum += len;
  }
  LOG(INFO) << "Net Params: " << sum << " Bytes @ " << net_params.size();
  return sum;
}

template <typename Dtype>
void Worker<Dtype>::sync(CDataRef data) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Worker<Dtype>::signal(CDataRef data) {
  NOT_IMPLEMENTED;
}


template <typename Dtype>
ParentWorker<Dtype>::ParentWorker(int children_size, const int *children,
    int data_size, char *memory)
  : Worker<Dtype>(), children_size_(children_size), data_size_(data_size)
  , children_(children), memory_(memory)
{
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  sigprocmask(SIG_BLOCK, &wait_set, NULL);
  
  set_for_clean(sizeof(Dtype), this);
  ::signal(SIGHUP, do_exit);
  ::signal(SIGINT, do_exit);
  ::signal(SIGTERM, do_exit);
  ::signal(SIGQUIT, do_exit);
  ::atexit(clean_at_exit);
  
  Caffe::set_mode(Caffe::CPU); // TODO: give some GPU resources
}

template <typename Dtype>
void ParentWorker<Dtype>::sync(CDataRef data) {
  int sig, children_ready_num = 0;
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  for (; ; ) {
    if (0 != sigwait(&wait_set, &sig)) {
      continue;
    }
    if (sig != SIGSYNC) {
      continue;
    }
    ++children_ready_num;
    if (children_ready_num >= children_size_ && check_all_child()) {
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
  worker->status = WorkerData::WORKING;
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
  worker->status = WorkerData::SYNCING;

  union sigval rc_val;
  rc_val.sival_int = 2;
  for (int i = 0; i < children_size_; i++) {
    const pid_t pid = children_[i];
    sigqueue(pid, SIGSYNC, rc_val);
  }
}

template <typename Dtype>
bool ParentWorker<Dtype>::check_all_child() {
  const WorkerData *worker = (const WorkerData *)memory_;
  for (int i = 0; i < children_size_; i++) {
    if (worker->status != WorkerData::SYNCING) {
      return false;
    }
    worker = worker->next(data_size_);
  }
  return true;
}

template <typename Dtype>
void ParentWorker<Dtype>::clean() {
  if (memory_ == NULL || data_size_ == 0) {
    return;
  }
  union sigval rc_val;
  rc_val.sival_int = -2;
  for (int i = 0; i < children_size_; i++) {
    const pid_t pid = children_[i];
    sigqueue(pid, SIGTERM, rc_val);
  }
  int msize = data_size_ * children_size_;
  if (munmap(memory_, msize) != 0) {
    LOG(ERROR) << "Release shared memory: fail: " << errno << " @ s=" << msize;
  }
}

template <typename Dtype>
void ParentWorker<Dtype>::setInterface(Interface &interface) {
  interface.setWorkerType(Interface::PARENT);
  interface.setChildIndex(0);
}


template <typename Dtype>
ChildWorker<Dtype>::ChildWorker(int child_index, int parent_pid,
    int data_size, char *memory, const char *parent_memory)
  : Worker<Dtype>(), child_index_(child_index), parent_pid_(parent_pid)
  , data_size_(data_size), memory_(memory), parent_memory_(parent_memory)
{
  WorkerData *worker = (WorkerData *)memory_;
  worker->status = WorkerData::WORKING;
  worker->pid = getpid();

  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  sigprocmask(SIG_BLOCK, &wait_set, NULL);
  
  LOG(INFO) << "Fork a child #" << child_index << ", map: " << (int*)memory;
  if (Caffe::mode() == Caffe::GPU) {
    int device_id = MPI::GetDevice(child_index);
    Caffe::SetDevice(device_id);
    LOG(INFO) << "Child #" << child_index << " use the device #" << device_id;
  }
  ::atexit(at_child_exit);
}

template <typename Dtype>
void ChildWorker<Dtype>::sync(CDataRef data) {
  WorkerData *worker = (WorkerData *)memory_;
  BufferUnit *buffer = worker->data;
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    const Dtype *diff_ptr = data[i]->cpu_diff();
    memcpy(buffer, diff_ptr, sizeof(Dtype) * count);
    buffer = buffer->next(count);
  }
  worker->status = WorkerData::SYNCING;

  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);

  union sigval rc_val;
  rc_val.sival_int = 1;
  sigqueue(parent_pid_, SIGSYNC, rc_val);
  for (int sig; ; ) {
    if (0 != sigwait(&wait_set, &sig)) {
      continue;
    }
    if (sig == SIGSYNC) {
      break;
    }
  }
  worker->status = WorkerData::WORKING;
  work(data);
}

template <typename Dtype>
void ChildWorker<Dtype>::work(CDataRef data) {
  volatile const WorkerData *const parent_worker =
      (volatile const WorkerData *)parent_memory_;
  volatile const BufferUnit *parent_buffer = parent_worker->data;
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    Dtype *data_ptr = data[i]->mutable_cpu_data();
    memcpy(data_ptr, (Dtype *)parent_buffer->data, sizeof(Dtype) * count);
    parent_buffer = parent_buffer->nextv(count);
  }
}

template <typename Dtype>
void ChildWorker<Dtype>::setInterface(Interface &interface) {
  interface.setWorkerType(Interface::CHILD);
  interface.setChildIndex(child_index_);
}

INSTANTIATE_CLASS(ParentWorker);
INSTANTIATE_CLASS(ChildWorker);

template int Worker<float>::GetParamsSize
(CDataRef net_params);
template int Worker<double>::GetParamsSize
(CDataRef net_params);

}  // namespace mpi
}  // namespace caffe

using namespace caffe::mpi;
static ParentWorker<float > *s_parent_f = NULL;
static ParentWorker<double> *s_parent_d = NULL;

static void set_for_clean(int type_size, void *instance) {
  if (type_size == 4) {
    s_parent_f = (ParentWorker<float >*)instance;
  } else if (type_size == 8) {
    s_parent_d = (ParentWorker<double>*)instance;
  }
}

void clean_at_exit() {
  if (s_parent_f != NULL) {
    s_parent_f->clean();
    s_parent_f = NULL;
  }
  if (s_parent_d != NULL) {
    s_parent_d->clean();
    s_parent_d = NULL;
  }
}

void do_exit(int sig) {
  exit(1);
}

void at_child_exit() {
  LOG(INFO) << "Child #" << caffe::MPI::child_index() << " exit.";
}

