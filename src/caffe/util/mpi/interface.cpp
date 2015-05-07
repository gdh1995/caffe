#include <errno.h>
#include <sys/mman.h>
#include <cstdlib>

#include "caffe/util/mpi/interface.hpp"
#include "caffe/util/mpi/worker.hpp"

using std::vector;
using boost::shared_ptr;
using caffe::Blob;

namespace caffe {
namespace mpi {

Interface mpi;

Interface()
  : worker_type_(SELF_ONLY), data_partition_(1), model_partition_(1)
  , device_count(0), device_list(NULL)
{
}

bool Interface::check_for_fork() {
  const int copy = data_partition_ * model_partition_;
  if (copy <= 1) {
    return false;
  }
  if (Caffe::mode() == Caffe::GPU) {
    if (device_count_ >= copy) {
      device_count_ = copy;
    } else if (device_count_ > 1) {
      LOG(ERROR) << "Parallel: should give " << copy << "devices";
      return false;
    } else {
      int *new_list = new int [copy];
      for (int i = 0; i < copy; i++) {
        new_list[i] = i;
      }
      device_list_ = new_list;
    }
  } else {
    device_count_ = copy;
  }
  return true;
}

template <typename Dtype>
void *Interface::do_fork(SharedParamsRef<Dtype> net_params) const {
  const int child_mem_size = Worker::GetParamsSize(net_params);
  const int shared_mem_size = child_mem_size * (data_partition_ + 1);
  char *shared_mem = (char *)mmap(NULL, shared_mem_size, PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_ANON, -1, 0);
  if (shared_mem == MAP_FAILED) {
    LOG(ERROR) << "Map shared memory: failed!";
    // one GPU has been selected in Caffe::SetDevice
    return SELF_ONLY;
  }

  pid_t parent_id = getpid();
  int *children = new int[device_count_];
  for (int i = 0; i < device_count_; ) {
    pid_t child = ::fork();
    if (child < 0) {
      LOG(ERROR) << "Fork failed when creating child worker #" << i;
      continue; // TODO: now may go into a dead loop
    } else if (child == 0) {
      worker_type_ = CHILD;
      shared_mem += child_mem_size * (i + 1);
      return new ChildWorker(i, parent_id, child_mem_size, shared_mem);
    }
    children[i] = child;
    i++;
  }
  worker_type_ = PARENT;
  return new ParentWorker(device_count_, children, child_mem_size, shared_mem);
}

void Interface::setup_handler(WorkerType type, Handler *func, void *data) {
  if (func == NULL) {
    return;
  }
  int index;
  switch(type) {
  case SELF_ONLY: index = 0; break;
  case PARENT: index = 1; break;
  case CHILD: index = 2; break;
  default:
    LOG(ERROR) << "Unknown MPI handler type: " << type << " *" << func;
    return;
  }
  mpi.handlers_[index].push({func, data});
}

void Interface::trigger() {
  int index;
  switch(type) {
  case SELF_ONLY: index = 0; break;
  case PARENT: index = 1; break;
  case CHILD: index = 2; break;
  default:
    LOG(ERROR) << "Unknown MPI handler type: " << type << " *" << func;
    return;
  }
  volatile vector<Callback> &handler = handlers_[index];
  while (!handler.empty()) {
    HandlerWrapper wrapper = handler.pop();
    wrapper.func(wrapper.data);
  }
}
