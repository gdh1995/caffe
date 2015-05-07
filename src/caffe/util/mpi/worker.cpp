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

static void set_for_clean();
static void clean_at_exit();
static void do_exit();

#define SIGSYNC (SIGUSR2)

namespace caffe {
namespace mpi {

template <typename DType, typename Ctype>
int Worker::GetParamsSize(CDataRef net_params) {
  int sum = sizeof(int) * 2;
  for (int i = 0; i < net_params.size(); i++) {
    Blob<Dtype> *blob = net_params[i].get();
    int len = sizeof(Dtype) * blob->count();
    sum += len;
  }
  LOG(INFO) << "Net Params: " << sum << " Bytes @ " << net_params.size();
  return sum;
}

template <typename DType, typename Ctype>
void Worker::sync(CDataRef data) {
  NOT_IMPLEMENTED;
}

template <typename DType, typename Ctype>
void Worker::signal(CDataRef data) {
  NOT_IMPLEMENTED;
}

template <typename DType, typename Ctype>
static void Worker::InitBufferArray(BufferUnit *buffer, CDataRef data) {
  // for (int i = 0; i < data.size(); i++) {
    // buffer->count = data[i]->count();
    // buffer = buffer->next();
  // }
}

template <typename DType, typename Ctype>
ParentWorker::ParentWorker(int children_size, const int *children,
    int data_size, char *memory)
  : children_size_(children_size), children_(children)
  , data_size_(data_size), memory_(memory), buffer_inited_(false)
{
  WorkerData *worker = (WorkerData *)memory_;
  worker->status = WORKING;
  worker->pid = getpid();

  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  sigprocmask(SIG_BLOCK, &wait_set, NULL);
  
  set_for_clean(sizeof(DType), this);
  signal(SIGINT, do_exit);
  signal(SIGTERM, do_exit);
  signal(SIGHUP, do_exit);
  atexit(clean_at_exit);
  
  Caffe::set_mode(Caffe::CPU); // TODO: give some GPU resources
  LOG(INFO) << "Shared memory: " << children_size + 1 << " * " << buffer_size;
}

template <typename DType, typename Ctype>
void ParentWorker::sync(CDataRef data) {
  WorkerData *worker = (const WorkerData *)memory_;
  if (!buffer_inited_) {
    InitBufferArray(worker->data, data);
    buffer_inited_ = true;
  }
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
  worker->status = WORKING;
  work(data);
}

template <typename DType, typename Ctype>
void ParentWorker::work(CDataRef data) {
  WorkerData *worker = (WorkerData *)memory_;
  BufferUnit *buffer = worker->data;
  const WorkerData *child_worker = worker->next(data_size_);
  const BufferUnit *child_buffer = child_worker->data;
  const int count = (children_size_ - 2 * sizeof(int)) / sizeof(DType);
  caffe_copy(count, (const DType *)child_buffer, (DType *)buffer);
  for (int i = 1; i < children_size_; i++) {
    child_worker = child_worker->next(data_size_);
    child_buffer = child_worker->data;
    caffe_axpy(count, (Dtype)1, (const Dtype *)child_buffer, (Dtype *)buffer);
  }
  caffe_scal(count, (Dtype)1 / children_size_, (Dtype *)buffer);

  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    DType *diff_ptr = data[i]->mutable_cpu_diff();
    caffe_copy(count, (const Dtype *)buffer, diff_ptr);
    buffer = buffer->next(count);
  }
}

template <typename DType, typename Ctype>
void ParentWorker::signal(CDataRef data) {
  WorkerData *worker = (WorkerData *)memory_;
  BufferUnit *buffer = worker->data;
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    const DType *cpu_data = data[i]->cpu_data();
    caffe_copy(count, cpu_data, (Dtype *)buffer);
    buffer = buffer->next(count);
  }
  worker->status = SYNCING;
  
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

template <typename DType, typename Ctype>
void ParentWorker::check_all_child() {
  const WorkerData *worker = (const WorkerData *)memory_;
  for (int i = 0; i < children_size; i++) {
    worker = worker->next(data_size_);
    if (worker->status != SYNCING) {
      return false;
    }
  }
  return true;
}

template <typename DType, typename Ctype>
void ParentWorker::clean() {
  if (memory_ = NULL || buffer_size_ == 0) {
    return;
  }
  int msize = buffer_size_ * (1 + children_size_);
  if (munmap(memory_, msize) != 0) {
    LOG(ERROR) << "Release shared memory: fail: " << errno << " @ s=" << msize;
  }
}


template <typename DType, typename Ctype>
ChildWorker::ChildWorker(int child_index, int parent_pid, int data_size,
    char *memory, const char *parent_memory)
  : child_index_(child_index), parent_pid_(parent_pid), data_size_(data_size)
  , memory_(memory), parent_memory_(parent_memory)
{
  WorkerData *worker = (WorkerData *)memory_;
  worker->status = WORKING;
  worker->pid = getpid();

  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  sigprocmask(SIG_BLOCK, &wait_set, NULL);
  
  LOG(INFO) << "Fork a child #" << child_index << ", map: " << memory;
  if (Caffe::mode() == Caffe::GPU) {
    int device_id = MPI::GetDevice(child_index);
    LOG(INFO) << "Child #" << child_index << " use the device #" << device_id;
    Caffe::SetDevice(device_id);
  }
}

template <typename DType, typename Ctype>
void ChildWorker::sync(CDataRef data) {
  static bool _inited = false;
  WorkerData *worker = (WorkerData *)memory_;
  BufferUnit *buffer = worker->data;
  if (!_inited) {
    InitBufferArray(buffer, data);
    _inited = true;
  }
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    const DType *diff_ptr = data[i]->cpu_diff();
    caffe_copy(count, cpu_data, (DType *)buffer);
    buffer = buffer->next(count);
  }
  worker->status = SYNCING;

  int sig;
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);

  volatile WorkerData *const parent_worker = (WorkerData *)parent_memory_;
  union sigval rc_val;
  rc_val.sival_int = 1;
  sigqueue(parent_id_, SIGSYNC, rc_val);
  for (; ; ) {
    sigwait(&wait_set, &sig);
    if (sig == SIGSYNC && parent_worker->status == SYNCING) {
      break;
    }
  }
  DLOG(INFO) << "Child #" << child_index_ << ": get merged data";
  worker->status = WORKING;
  work(data);
}

template <typename DType, typename Ctype>
void ChildWorker::signal(CDataRef data) {
  volatile WorkerData *const parent_worker = (WorkerData *)parent_memory_;
  volatile BufferUnit *parent_buffer = parent_worker->data;
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count();
    DType *data_ptr = data[i]->mutable_cpu_data();
    caffe_copy(count, (const DType *)parent_buffer, data_ptr);
    parent_buffer = parent_buffer->next(count);
  }
}

INSTANTIATE_CLASS(ParentWorker);
INSTANTIATE_CLASS(ChildWorker);
  
}  // namespace mpi
}  // namespace caffe

using namespace caffe::mpi;
static ParentWorker<float , vector<shared_ptr<Blob> > > *s_parent_f = NULL;
static ParentWorker<double, vector<shared_ptr<Blob> > > *s_parent_d = NULL;

static void set_for_clean(int type_size, void *instance) {
  if (type_size == 4) {
    s_parent_f = (ParentWorker<float , vector<shared_ptr<Blob> > >*)instance;
  } else if (type_size == 8) {
    s_parent_d = (ParentWorker<double, vector<shared_ptr<Blob> > >*)instance;
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

void do_exit() {
  exit();
}