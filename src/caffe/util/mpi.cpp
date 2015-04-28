#include <glog/logging.h>
#include <cstdio>
#include <ctime>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <errno.h>
#include <sys/mman.h>
#include <signal.h>

#include "caffe/util/mpi.hpp"

using std::vector;
using boost::shared_ptr;
using caffe::Blob;
using caffe::MPInterface;

static int device_count = 1;
static const int * device_list = NULL;

static pid_t parent_id = 0;
static void *shared_mem = NULL;
static int shared_mem_size = 0;
static int child_mem_size = 0;
static int child_ready_count = 0;

static vector<MPInterface::Callback> onforks;
static vector<MPInterface::Callback> on_not_forks;
static vector<void *> onfork_data;

static void initChild(void);
static void unfork(void);
static void mem_from(int i, int len, const void *data);

static void initParent();
static void set_child_ready();
static bool check_all_child();
static void signalChildren();
static void cleanNexti();

template <typename Dtype>
static void work_child(const vector<shared_ptr<Blob<Dtype> > > &data);
template <typename Dtype>
static void work_parent(const vector<shared_ptr<Blob<Dtype> > > &data);

namespace caffe {

void Caffe::SetDevice(const int *id_list, const int count) {
#ifndef CPU_ONLY  // Normal GPU + CPU Caffe.
  int last = 0;
  if (count < 1) {
    device_count = 0;
    return;
  }
  int *new_list = new int [count];
  for (int i = 0; i < count; i++) {
    const int id = id_list[i];
    if (id >= 0) {
      cudaError_t error = cudaSetDevice(id);
      if (error == cudaSuccess) {
        new_list[last] = id;
        last++;
      }
    }
  }
  device_count = last;
  if (last <= 0) {
    LOG(ERROR) << "Devices: all are invalid";
  } else if (last == 1) {
    cudaSetDevice(-1); // TODO: test and remove
    const int new_id = new_list[0];
// #ifdef _DEBUG
    int current_device = -1;
    cudaGetDevice(&current_device);
    if (current_device == new_id) {
      LOG(INFO) << "Devices: Get a device set but not inited: " << new_id;
    }
// #endif
    SetDevice(new_id);
  } else {
    device_list = new_list;
  }
#else  // CPU-only Caffe.
  NO_GPU;
#endif
}

int MPInterface::data_partition_ = 1;
int MPInterface::model_partition_ = 1;

MPInterface::ForkStatus MPInterface::fork_stat_ = MPInterface::NONE;
int MPInterface::child_index_ = 0;

void MPInterface::set_copy(const int data_copy, const int model_copy) {
  CHECK_GE(data_copy, 1) << "Copy number is invalid.";
  data_partition_ = data_copy;
  CHECK_GE(model_copy, 1) << "Copy number is invalid.";
  model_partition_ = model_copy;
  // TODO: check batch size & split data layer
}

void MPInterface::setup_onfork(Callback func, void *data, Callback err_func) {
  onforks.push_back(func);
  on_not_forks.push_back(err_func);
  onfork_data.push_back(data);
}

template <typename Dtype>
void MPInterface::calc_shared_mem(const vector<shared_ptr<Blob<Dtype> > > &net_params) {
  if (data_partition_ <= 1) {
    return;
  }
  int sum = 0;
  for (int i = 0; i < net_params.size(); i++) {
    Blob<Dtype> *blob = net_params[i].get();
    int len = sizeof(Dtype) * blob->count();
    len = (len + 15) / 16;
    sum += sizeof(int) * 4 + len * 16;
  }
  child_mem_size = sizeof(int) * 4 + sum;
  shared_mem_size = child_mem_size * (data_partition_ + 1);
  LOG(INFO) << "ChildUnit: " << child_mem_size << " @ " << net_params.size();
}

template
void MPInterface::calc_shared_mem(const vector<shared_ptr<Blob<float> > > &net_params);
template
void MPInterface::calc_shared_mem(const vector<shared_ptr<Blob<double> > > &net_params);

MPInterface::ForkStatus MPInterface::do_fork() {
  if (data_partition_ <= 1 && model_partition_ <= 1) {
    return NONE;
  }
  const int copy = data_partition_ * model_partition_;
  if (Caffe::mode() == Caffe::GPU) {
    if (device_count >= copy) {
      device_count = copy;
    } else if (device_count > 1) {
      LOG(ERROR) << "Devices: should give " << copy << "device ids";
      return NONE;
    } else {
      int *new_list = new int [copy];
      for (int i = 0; i < copy; i++) {
        new_list[i] = i;
      }
      device_list = new_list;
    }
  } else {
    device_count = copy;
  }
  shared_mem = mmap(NULL, shared_mem_size, PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_ANON, -1, 0);
  if (shared_mem == MAP_FAILED) {
    LOG(ERROR) << "Map shared memory: failed!";
    if (Caffe::mode() == Caffe::GPU) {
      Caffe::SetDevice(device_list[0]); // select the first device and use this only
    }
    shared_mem = NULL;
    return NONE;
  }

  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGUSR2);
  sigprocmask(SIG_BLOCK, &wait_set, NULL);

  parent_id = getpid();
  for (int i = 0, size = device_count; i < size; ) {
    pid_t child = ::fork();
    if (child < 0) {
      LOG(ERROR) << "Fork failed when creating #" << (i + 1);
      continue; // TODO: may go into a dead loop
    } else if (child == 0) {
      child_index_ = i;
      initChild();
      return CHILD;
    }
    i++;
    // TODO: remember info;
  }
  
  initParent();
  return PARENT;
}

void MPInterface::trigger() {
  const vector<Callback> &handlers = fork_stat_ == CHILD ? onforks : on_not_forks;
  for (int i = 0; i < handlers.size(); i++) {
    if (handlers[i] != NULL) {
      handlers[i](onfork_data[i]);
    }
  }
}

template <> void MPInterface::sync<unsigned int>
(const vector<shared_ptr<Blob<unsigned int> > > &data) {
  NOT_IMPLEMENTED;
}

template <> void MPInterface::sync<int>
(const vector<shared_ptr<Blob<int> > > &data) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MPInterface::sync(const vector<shared_ptr<Blob<Dtype> > > &data) {
  if (fork_stat() == NONE) {
    return;
  }
  if (fork_stat() == CHILD) {
    cleanNexti();
    for (int i = 0; i < data.size(); i++) {
      Blob<Dtype> *blob = data[i].get();
      mem_from(i, sizeof(Dtype) * blob->count(), blob->cpu_diff());
    }
    set_child_ready();
    union sigval rc_val;
    rc_val.sival_int = 1;
    sigqueue(parent_id, SIGUSR2, rc_val);
  }
  int sig;
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGUSR2);
  for (sig = 0; ; ) {
    sigwait(&wait_set, &sig);
    if (sig != SIGUSR2) {
      continue;
    } 
    if (fork_stat() == CHILD) {
      LOG(INFO) << "Child #" << child_index_ << ": get merged data";
      work_child(data);
      return;
    }
    ++child_ready_count;
    LOG(INFO) << "Get SIGUSR2, count = " << child_ready_count;
    if (child_ready_count >= device_count && check_all_child()) {
      LOG(INFO) << "All children are waiting to sync.";
      LOG(INFO) << "Parent #" << child_index_ << ": " << (fork_stat() == PARENT);
      child_ready_count = 0;
      work_parent(data);
      return;
    }
  }
}

template
void MPInterface::sync(const vector<shared_ptr<Blob<float> > > &data);
template
void MPInterface::sync(const vector<shared_ptr<Blob<double> > > &data);

template <typename Dtype>
void MPInterface::signal(const vector<shared_ptr<Blob<Dtype> > > &data) {
  cleanNexti();
  for (int i = 0; i < data.size(); i++) {
    Blob<Dtype> *blob = data[i].get();
    mem_from(i, sizeof(Dtype) * blob->count(), blob->cpu_data());
  }
  signalChildren();
}

template
void MPInterface::signal(const vector<shared_ptr<Blob<float> > > &data);
template
void MPInterface::signal(const vector<shared_ptr<Blob<double> > > &data);

}  // namespace caffe

using namespace caffe;

struct ChildUnit {
  enum Status {ITER, SYNC};
  int nexti, status, pid, mask3;
  char data[0]; // [i, len, mask, mask, ...]
};

ChildUnit *child;

void cleanNexti() {
  child->nexti = 0;
}

void initChild() {
  child = (ChildUnit *)((char *)shared_mem + child_mem_size * (MPI::child_index() + 1));
  child->nexti = 0;
  child->status = ChildUnit::ITER;
  child->pid = getpid();
  LOG(INFO) << "Fork a child #" << MPI::child_index() << ", map: " << child;
  if (Caffe::mode() == Caffe::GPU) {
    LOG(INFO) << "Child #" << MPI::child_index() << " use the device #"
        << device_list[MPI::child_index()];
    Caffe::SetDevice(device_list[MPI::child_index()]);
  }
}

void initParent() {
  child = (ChildUnit *)shared_mem;
  child->nexti = 0;
  child->status = ChildUnit::SYNC;
  child->pid = parent_id;

  atexit(unfork);

  Caffe::set_mode(Caffe::CPU); // TODO: give some GPU resources
  LOG(INFO) << "Fork finished. Pid of the parent is " << parent_id;
  LOG(INFO) << "Shared memory: " << child << " + " << shared_mem_size;
}

void unfork() {
  if (MPI::fork_stat() == MPI::PARENT) {
    if (shared_mem != NULL) {
      munmap(shared_mem, shared_mem_size);
      shared_mem = NULL;
    }
  }
}

void set_child_ready() {
  child->status = ChildUnit::SYNC;
}

bool check_all_child() {
  const char *p = (const char *)shared_mem;
  for (int i = 0; i < device_count; i++) {
    p += child_mem_size;
    const ChildUnit * const child = (const ChildUnit *)p;
    LOG(INFO) << "child #" << child << ", pid #" << child->pid
        << " is " << child->status << " with nexti=" << child->nexti;
    if (child->status != ChildUnit::SYNC) {
      return false;
    }
  }
  return true;
}

void signalChildren() {
  union sigval rc_val;
  rc_val.sival_int = 2;
  const char *p = (const char *)shared_mem;
  std::ostringstream oss;
  oss << "Parent: send signal to";
  for (int i = 0; i < device_count; i++) {
    p += child_mem_size;
    const ChildUnit * const child = (const ChildUnit *)p;
    oss << " " << child->pid;
    sigqueue(child->pid, SIGUSR2, rc_val);
  }
  LOG(INFO) << oss.str();
}

void mem_from(int i, int len, const void *data) {
  int *blob = (int *)(child->data + child->nexti * 4);
  blob[0] = i, blob[1] = len;
  memcpy(blob + 4, data, len);
  len = (len + 15) / 16;
  child->nexti += 4 + len * 4;
}

void mem_to(int i, int len, void *data, const char * const from_data, int &nexti) {
  const int * blob = (int *)(from_data + nexti * 4);
  CHECK_EQ(i, blob[0]) << "Net.Params[i] has an error index in s-mem.";
  CHECK_EQ(len, blob[1]) << "Net.Params[i] has no same shape with s-mem.";
  memcpy(data, blob + 4, len);
  len = (len + 15) / 16;
  nexti += 4 + len * 4;
}


template <typename Dtype>
static void work_child(const vector<shared_ptr<Blob<Dtype> > > &data) {
  const char * parent_data = ((ChildUnit *)shared_mem)->data;
  int nexti = 0;
  for (int i = 0; i < data.size(); i++) {
    const int len = data[i]->count() * sizeof(Dtype);
    mem_to(i, len, data[i]->mutable_cpu_data(), parent_data, nexti);
  }
}

template <typename Dtype>
static void work_parent(const vector<shared_ptr<Blob<Dtype> > > &data) {
  CHECK_EQ(shared_mem, child) << "parent's shared block is not \"index == 0\".";
  caffe_copy(child_mem_size / sizeof(int) - 4
      , (int*)(child->data + child_mem_size)
      , (int*)child->data);
  child->nexti = 0;
  int *blob;
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count(), len = count * sizeof(Dtype);
    blob = (int *)(child->data + child->nexti * 4);
    // LOG(INFO) << "Diff will sum to " << blob << " + " << len;
    CHECK_EQ(i, blob[0]) << "Net.params.diff_blob.index() error.";
    CHECK_EQ(len, blob[1]) << "Net.params.diff_blob.count() error.";
    const int *blob_j = (const int *)(((const char*)blob) + child_mem_size);
    for (int j = 1; j < device_count; j++) {
      blob_j = (const int *)(((const char*)blob_j) + child_mem_size);
      CHECK_EQ(len, blob_j[1]) << "diff_blob.count() does not match.";
      caffe_axpy(count, (Dtype)1, (const Dtype *)(blob_j + 4)
          , (Dtype *)(blob + 4));
    }
    caffe_scal(count, ((Dtype)1) / ((float)device_count)
          , (Dtype *)(blob + 4));
    mem_to(i, len, data[i]->mutable_cpu_diff(), child->data, child->nexti);
  }
}
