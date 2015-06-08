#include "caffe/util/mpi/worker.hpp"
#include "caffe/blob.hpp"
#include <signal.h>
#include <pthread.h>

namespace caffe {
namespace mpi {

const int SIGSYNC = SIGRTMIN + 1;

static pthread_t main_thread_id;
static void forward_signal(int sig);

void block_signal_for_sync() {
  sigset_t wait_set;
  sigemptyset(&wait_set);
  sigaddset(&wait_set, SIGSYNC);
  pthread_sigmask(SIG_BLOCK, &wait_set, NULL);
  // for other threads, forward to 
  main_thread_id = pthread_self();
  ::signal(SIGSYNC, forward_signal);
}

void forward_signal(int sig) {
  union sigval rc_val;
  rc_val.sival_int = 1;
  pthread_sigqueue(main_thread_id, sig, rc_val);
  DLOG(INFO) << "MPI: this thread handled signal " << sig;
}

int get_parent_device_id() {
  return MPI::GetDevice(0);
}

int set_peer_device(int peer_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == peer_id) {
    return 0;
  }
  int canAccessPeer = -1;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, current_device, peer_id));
  LOG(INFO) << "check direct peer access: " << canAccessPeer << " (1 := capable)";
  cudaError_t err = cudaDeviceEnablePeerAccess(peer_id, 0);
  if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
    return 0;
  } else {
    LOG(ERROR) << "can not access peer device " << peer_id << " @ err: " << err;
    CUDA_CHECK(err);
    return -1;
  }
}

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
SelfWorker<Dtype>::SelfWorker() : Worker<Dtype>() {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  Caffe::SetDevice(current_device);
}

template int Worker<float>::GetParamsSize(CDataRef net_params);
template int Worker<double>::GetParamsSize(CDataRef net_params);
template SelfWorker<float>::SelfWorker();
template SelfWorker<double>::SelfWorker();


}  // namespace mpi
}  // namespace caffe
