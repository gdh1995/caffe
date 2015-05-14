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
{
}

bool Interface::check_for_fork() {
  const int copy = data_partition_ * model_partition_;
  if (copy <= 1) {
    return false;
  }
  if (Caffe::mode() == Caffe::GPU && device_count_ < copy) {
    if (device_count_ > 1) {
      LOG(ERROR) << "Parallel: should give " << copy << "devices";
    }
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
void *Interface::do_fork(const vector<shared_ptr<Blob<Dtype> > > &net_params) const {
  const int child_mem_size = Worker<Dtype>::GetParamsSize(net_params);
  const int shared_mem_size = child_mem_size * data_partition_;
  LOG(INFO) << "Shared memory: " << data_partition_ << " * " << child_mem_size;
  char *const shared_mem = (char *)mmap(NULL, shared_mem_size,
      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
  if (shared_mem == MAP_FAILED) {
    LOG(ERROR) << "Map shared memory: failed!";
    // one GPU has been selected in Caffe::SetDevice
    return new SelfWorker<Dtype>();
  }

  pid_t parent_id = getpid();
  int *children = new int[device_count_];
  for (int i = 0; i < device_count_; ) {
    pid_t child = ::fork();
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
  return new ParentWorker<Dtype>(device_count_, children, child_mem_size, shared_mem);
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
  vector<HandlerWrapper> &handler = handlers_[index];
  while (!handler.empty()) {
    HandlerWrapper wrapper = handler.back();
    handler.pop_back();
    wrapper.func(wrapper.data);
  }
}


Interface Interface::mpi;
template void *Interface::do_fork
(const vector<shared_ptr<Blob<float> > > &net_params) const;
template void *Interface::do_fork
(const vector<shared_ptr<Blob<double> > > &net_params) const;

template class BaseWorker<float>;
template class BaseWorker<double>;


}  // namespace mpi
}  // namespace caffe
