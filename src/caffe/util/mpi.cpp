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

#include "caffe/common.hpp"
#include "caffe/util/mpi.hpp"

static int device_count = 1;
static const int * device_list = NULL;

static pid_t parent_id = 0;
static void *shared_mem = NULL;
static int shared_mem_size = 0;
static int child_mem_size = 0;
static int child_ready_count = 0;

static vector<caffe::MPInterface::Callback> onforks;
static vector<void *> onfork_data;

static void initChild(void);
static void unfork(void);
static void mem_from(int i, int len, const void *data);

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
    cudaError_t error = cudaGetDevice(&current_device);
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

ForkStatus MPInterface::fork_stat_ = MPInterface::ForkStatus::NONE;
int MPInterface::child_index_ = 0;

void MPInterface::set_copy(const int data_copy, const int model_copy) {
  CHECK_GE(data_copy, 1) << "Copy number is invalid.";
  data_partition_ = data_copy;
  CHECK_GE(model_copy, 1) << "Copy number is invalid.";
  model_partition_ = model_copy;
  // TODO: check batch size & split data layer
}

void MPInterface::setup_onfork(Callback func, void *data) {
  onforks.push(func);
  onfork_data.push(data);
}

MPInterface::ForkStatus MPInterface::do_fork() {
  if (data_partition_ <= 1 && model_partition_ <= 1) {
    return NONE;
  }
  const int copy = data_partition_ * model_partition_;
  if (Caffe::mode() == Caffe::GPU) {
    if (device_count >= copy) {
      device_count = copy;
    } else if (device_count >= 1) {
      LOG(ERROR) << "Devices: should give " << copy << "device ids";
      return NONE;
    } else {
      int *new_list = new int [copy];
      for (int i = 0; i < copy; i++) {
        new_list[i] = i;
      }
      device_list = new_list;
    }
  }
  shared_mem = mmap(NULL, shared_mem_size, PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_ANON, -1, 0);
  if (shared_mem == MAP_FAILED) {
    LOG(ERROR) << "Map shared memory: failed!";
    if (Caffe::mode() == Caffe::GPU) {
      SetDevice(device_list[0]); // select the first device and use this only
    }
    shared_mem = NULL;
    return NONE;
  }

  parent_id = getpid();
  for (int i = 0, size = device_count; i < size; ) {
    pid_t child = fork();
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
  
  void initParent();
  initParent();
  return PARENT;
}

template <> void MPInterface::sync<unsigned int>
(const vector<shared_ptr<Blob<Dtype> > > &data) {
  NOT_IMPLEMENTED;
}

template <> void MPInterface::sync<int>
(const vector<shared_ptr<Blob<Dtype> > > &data) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MPInterface::sync(const vector<shared_ptr<Blob<Dtype> > > &data) {
  if (fork_stat() == NONE) {
    return;
  }
  if (fork_stat() == CHILD) {
    for (int i = 0; i < data.size(); i++) {
      Blob<Dtype> *blob = data[i].get();
      mem_from(i, sizeof(Dtype) * blob->count(), blob->cpu_diff());
    }
    void set_child_ready();
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
      work_child(data);
      return;
    }
    // TODO: log info;
    ++child_ready_count;
    LOG(INFO) << "Get SIGUSR2, count = " << child_ready_count;
    bool check_all_child();
    if (child_ready_count >= device_count && check_all_child()) {
      LOG(INFO) << "All children are waiting to sync.";
      child_ready_count = 0;
      work_parent(data);
      return;
    }
  }
}

template <typename Dtype>
void MPInterface::signal(const vector<shared_ptr<Blob<Dtype> > > &data) {
  for (int i = 0; i < data.size(); i++) {
    Blob<Dtype> *blob = data[i].get();
    mem_from(i, sizeof(Dtype) * blob->count(), blob->cpu_data());
  }
  void signalChildren();
  signalChildren();
}

}  // namespace caffe

using namespace caffe;

struct ChildUnit {
  enum Status {ITER, SYNC};
  int nexti, status, pid, mask3;
  char data[0];
};

ChildUnit *child;

void initChild() {
  child = (ChildUnit *)(shared_mem + child_mem_size * (MPI::child_index() + 1));
  child->nexti = 0;
  child->status = ChildUnit::ITER;
  child->pid = getpid();
  LOG(INFO) << "Fork a child #" << MPI::child_index();
  if (Caffe::mode() == Caffe::GPU) {
    LOG(INFO) << "Child #" << MPI::child_index() << "user the device #"
        << device_list[MPI::child_index()];
    SetDevice(device_list[MPI::child_index()]);
  }
  for (int i = 0; i < onforks.size(); i++) {
    onforks[i](onfork_data[i]);
  }
}

void initParent() {
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGUSR2);
  sigprocmask(SIG_BLOCK, &procmask, NULL);

  child = (ChildUnit *)shared_mem;
  child->nexti = 0;
  child->status = SYNC;
  child->pid = parent_id;

  atexit(unfork);

  Caffe::set_mode(Caffe::CPU); // TODO: give some GPU resources
  LOG(INFO) << "Fork finished. Pid of the parent is " << parent_id;
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
  const char *p = shared_mem;
  for (int i = 0; i < device_count; i++) {
    p += child_mem_size;
    const ChildUnit *child = (const ChildUnit *)p;
    if (child->status != ChildUnit::SYNC) {
      return false;
    }
  }
  return true;
}

void signalChildren() {
  union sigval rc_val;
  rc_val.sival_int = 2;
  const char *p = shared_mem;
  for (int i = 0; i < device_count; i++) {
    p += child_mem_size;
    const ChildUnit *child = (const ChildUnit *)p;
    sigqueue(child->pid, SIGUSR2, rc_val);
  }
}

void mem_from(int i, int len, const void *data) {
  int *blob = (int *)(child->data + child->nexti * 4);
  blob[0] = i, blob[1] = len;
  memcpy(blob + 4, data, len);
  len = (len + 15) / 16;
  child->nexti += 4 + len * 4;
}

void mem_to(int i, int len, void *data) {
  const int *blob = (int *)(child->data + child->nexti * 4);
  CHECK_EQ(i, blob[0]) << "Net.Params[i] has an error index in s-mem.";
  CHECK_EQ(len, blob[1]) << "Net.Params[i] has no same shape with s-mem.";
  memcpy(data, blob + 4, len);
  len = (len + 15) / 16;
  child->nexti += 4 + len * 4;
}


template <typename Dtype>
static void work_child(const vector<shared_ptr<Blob<Dtype> > > &data) {
  child->nexti = 0;
  void *old_child = child;
  child = (ChildUnit *)shared_mem;
  int *blob;
  for (int i = 0; i < data.size(); i++) {
    const int len = data[i]->count() * sizeof(Dtype);
    mem_to(i, len, data[i]->mutable_cpu_data());
  }
  child = (ChildUnit *)old_child;
}

template <typename Dtype>
static void work_parent(const vector<shared_ptr<Blob<Dtype> > > &data) {
  child->nexti = 0;
  CHECK_EQ(shared_mem, child) << "parent's shared block is not \"index == 0\".";
  caffe_copy(child_mem_size / sizeof(int), (int*)(shared_mem + child_mem_size), (int*)child);
  int *blob;
  for (int i = 0; i < data.size(); i++) {
    const int count = data[i]->count(), len = count * sizeof(Dtype);
    blob = (int *)(child->data + child->nexti * 4);
    CHECK_EQ(i, blob[0]) << "Net.params.diff_blob.index() error.";
    CHECK_EQ(len, blob[1]) << "Net.params.diff_blob.count() error.";
    const int *blob_j = (const int *)(((const char*)blob) + child_mem_size);
    for (int j = 1; j < device_count; j++) {
      blob_j = (const int *)(((const char*)blob_j) + child_mem_size);
      CHECK_EQ(len, blob_j[1]) << "diff_blob.count() does not match.";
      caffe_add(count, (const Dtype *)(blob_j + 4)
          , (Dtype *)(blob + 4));
    }
    caffe_scal(count, ((Dtype)1) / ((float)device_count)
          , (Dtype *)(blob + 4));
    mem_to(i, len, data[i]->mutable_cpu_diff());
  }
}


// parent_id's message loop
void waitSignal() {
  sigset_t wait_set;
  int sig_no;
  const struct timespec tv = {5,0}; //timeout
  siginfo_t sig_info;
  pid_t pid = 0;

  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGUSR2);

  signal(SIGCHLD,SIG_IGN);    // avoid Zombie

  if(parent_id == 0) {
      int rc = 0;
      union sigval rc_val; // message passed
      setsid();
      rc = system("curl -O http://www.kernel.org/pub/linux/kernel/v2.4/linux-2.4.24.tar.gz &>/dev/null");
      printf("rc = %d\n", rc);
      rc_val.sival_int = rc/255;
      sigqueue(parent_id, SIGUSR2, rc_val);
      exit(0);
  }

  //parent_id process
  for (; ; ) {
    sigprocmask(&wait_set);
    sig_no = sigtimedwait(&wait_set, &sig_info, &tv);
    if(sig_no == -1) {
        if(errno == EINTR) {
            continue;
        } else if(errno == EAGAIN); // time out
            printf("child process timeout \n");
        }
        continue;
    }
    int msg = sig_info.si_int;
    if(msg == 0) {
      
    } else {
    }
  }
}
