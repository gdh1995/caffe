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

#define SIGSYNC (SIGUSR2)

namespace caffe {
namespace mpi {

template <typename Dtype>
int Worker<Dtype>::GetParamsSize(CDataRef net_params) {
  int sum = sizeof(int) * 2;
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
  WorkerData *worker = (WorkerData *)memory_;
  worker->status = WorkerData::WORKING;
  worker->pid = getpid();

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
  LOG(INFO) << "Shared memory: " << children_size + 1 << " * " << data_size_;
}

template <typename Dtype>
void ParentWorker<Dtype>::sync(CDataRef data) {
  WorkerData *worker = (WorkerData *)memory_;
  int sig, children_ready_num;
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  children_ready_num = 0;
  for (; ; ) {
    sigwait(&wait_set, &sig);
    if (sig != SIGSYNC) {
      continue;
    }
    ++children_ready_num;
    if (children_ready_num >= children_size_ && check_all_child()) {
      break;
    }
  }
  DLOG(INFO) << "All children are waiting to sync.";
  worker->status = WorkerData::WORKING;
  work(data);
}

template <typename Dtype>
void ParentWorker<Dtype>::work(CDataRef data) {
  WorkerData *worker = (WorkerData *)memory_;
  BufferUnit *buffer = worker->data;
  const WorkerData *child_worker = worker->next(data_size_);
  const BufferUnit *child_buffer = child_worker->data;
  const int count = (data_size_ - 2 * sizeof(int)) / sizeof(Dtype);
  //LOG(INFO) << "Parent working: ds = " << count;
  //LOG(INFO) << "  CP" << buffer << child_worker << child_buffer;
  caffe_copy(count, (const Dtype *)child_buffer, (Dtype *)buffer);
  for (int i = 1; i < children_size_; i++) {
    child_worker = child_worker->next(data_size_);
    child_buffer = child_worker->data;
    //LOG(INFO) << "  PX " << i << ": " << child_worker << child_buffer;
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
  
  std::ostringstream oss;
  oss << "Parent: send signal to";
  union sigval rc_val;
  rc_val.sival_int = 2;
  for (int i = 0; i < children_size_; i++) {
    const pid_t pid = children_[i];
    oss << " " << pid;
    sigqueue(pid, SIGSYNC, rc_val);
  }
  DLOG(INFO) << oss.str();
}

template <typename Dtype>
bool ParentWorker<Dtype>::check_all_child() {
  const WorkerData *worker = (const WorkerData *)memory_;
  for (int i = 0; i < children_size_; i++) {
    worker = worker->next(data_size_);
    if (worker->status != WorkerData::SYNCING) {
      return false;
    }
  }
  return true;
}

template <typename Dtype>
void ParentWorker<Dtype>::clean() {
  if (memory_ == NULL || data_size_ == 0) {
    return;
  }
  union sigval rc_val;
  rc_val.sival_int = 1;
  for (int i = 0; i < children_size_; i++) {
    const pid_t pid = children_[i];
    sigqueue(pid, SIGTERM, rc_val);
  }
  int msize = data_size_ * (1 + children_size_);
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
    LOG(INFO) << "Child #" << child_index << " use the device #" << device_id;
    Caffe::SetDevice(device_id);
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
    caffe_copy(count, diff_ptr, (Dtype *)buffer);
    buffer = buffer->next(count);
  }
  worker->status = WorkerData::SYNCING;

  int sig;
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);

  volatile const WorkerData *const parent_worker =
      (volatile const WorkerData *)parent_memory_;
  union sigval rc_val;
  rc_val.sival_int = 1;
  sigqueue(parent_pid_, SIGSYNC, rc_val);
  for (; ; ) {
    sigwait(&wait_set, &sig);
    if (sig == SIGSYNC && parent_worker->status == WorkerData::SYNCING) {
      break;
    }
  }
  DLOG(INFO) << "Child #" << child_index_ << ": get merged data";
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
    caffe_copy(count, (const Dtype *)parent_buffer, data_ptr);
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

