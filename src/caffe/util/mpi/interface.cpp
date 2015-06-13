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

Interface::Interface()
  : worker_type_(SELF_ONLY), child_index_(0), data_partition_(1)
  , model_partition_(1), device_count_(0), device_list_(NULL)
  , shared_host_memory_(NULL), shared_host_mem_size_(0)
{
}

Interface::~Interface() {
  delete [] device_list_;
  if (shared_host_mem_size_ > 0 && shared_host_memory_ != NULL) {
    if (munmap(shared_host_memory_, shared_host_mem_size_) != 0) {
      LOG(ERROR) << "Release shared memory: fail: " << errno << " @ s="
          << shared_host_mem_size_;
    } else {
      LOG(INFO) << "Release shared memory: " << shared_host_mem_size_;
    }
  }
}

bool Interface::check_for_fork() {
  const int copy = data_partition_;
  if (copy <= 1) {
    return false;
  }
  if (Caffe::mode() == Caffe::GPU && device_count_ < copy) {
    CHECK_LE(device_count_, 1) << "Parallel: should give " << copy << "devices";
    int *new_list = new int [copy];
    for (int i = 0; i < copy; i++) {
      new_list[i] = i;
    }
    device_list_ = new_list;
  }
  device_count_ = copy;
  return true;
}

template <typename Dtype>
SafeClass *Interface::do_fork(
    const vector<shared_ptr<Blob<Dtype> > > *net_params) const {
  if (net_params == NULL) {
    return new SelfWorker<Dtype>();
  }

  const int fork_count = data_partition_;
  const int child_mem_size = Worker<Dtype>::GetParamsSize(*net_params);
  const int shared_mem_size = child_mem_size * fork_count;
  LOG(INFO) << "Shared memory: " << fork_count << " * " << child_mem_size;
  char *const shared_mem = (char *)mmap(NULL, shared_mem_size,
      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
  if (shared_mem == MAP_FAILED) {
    LOG(ERROR) << "Map shared memory: failed!";
    // one GPU has been selected in Caffe::SetDevice
    return new SelfWorker<Dtype>();
  }

  const pid_t parent_id = getpid();
  int *children = new int[fork_count];
  for (int i = 0; i < fork_count; ) {
    const pid_t child = ::fork();
    if (child < 0) {
      LOG(ERROR) << "Fork failed when creating child worker #" << i;
      continue; // TODO: now may go into a dead loop
    } else if (child == 0) {
      char *self_mem = shared_mem + child_mem_size * i;
      return new ChildWorker<Dtype>(i, parent_id, child_mem_size, self_mem,
          shared_mem);
    }
    children[i] = child;
    i++;
  }
  return new ParentWorker<Dtype>(fork_count, children, child_mem_size, shared_mem);
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
  HandlerWrapper wrapper = {func, data};
  mpi.handlers_[index].push_back(wrapper);
}

void Interface::triggerHandlers() {
  int index;
  switch(worker_type_) {
  case SELF_ONLY: index = 0; break;
  case PARENT: index = 1; break;
  case CHILD: index = 2; break;
  default:
    LOG(ERROR) << "Unknown MPI handler type: " << worker_type_;
    return;
  }
  const vector<HandlerWrapper> &handler = handlers_[index];
  for (int i = 0; i < handler.size(); i++) {
    const HandlerWrapper &wrapper = handler[i];
    wrapper.func(wrapper.data);
  }
  vector<HandlerWrapper>(0).swap(handlers_[0]);
  vector<HandlerWrapper>(0).swap(handlers_[1]);
  vector<HandlerWrapper>(0).swap(handlers_[2]);
}


Interface Interface::mpi;
template SafeClass *Interface::do_fork
(const vector<shared_ptr<Blob<float> > > *net_params) const;
template SafeClass *Interface::do_fork
(const vector<shared_ptr<Blob<double> > > *net_params) const;

template class BaseWorker<float>;
template class BaseWorker<double>;


}  // namespace mpi
}  // namespace caffe
