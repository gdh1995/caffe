#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/syncedmem.hpp"
#include <stdlib.h>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using caffe::SyncedMemory;
using caffe::InternalThread;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_int32(peer, -1,
    "Set peer to given device ID.");
    
// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);


SyncedMemory *shared_gmem;
const int shared_size = 4096;

class ThreadPeer : public InternalThread {
 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {
    Caffe::SetDevice(FLAGS_peer);
    SyncedMemory peer_gmem(shared_size);
    CUDA_CHECK(cudaMemset(peer_gmem.mutable_gpu_data(), 0xfa, shared_size));
    CUDA_CHECK(cudaStreamSynchronize(0));
    while(!shared_gmem) {}
    
    int canAccessPeer = -1;
    LOG(INFO) << "Peer: find target: " << FLAGS_gpu << " from " << FLAGS_peer;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, FLAGS_peer, FLAGS_gpu));
    LOG(INFO) << "Check direct peer access: " << canAccessPeer << " @1=capable";
    cudaError_t err = cudaDeviceEnablePeerAccess(FLAGS_gpu, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      LOG(ERROR) << "Can not access peer device " << FLAGS_gpu << " @e=" << err;
      CUDA_CHECK(err);
      return;
    }
    LOG(INFO) << "Access to Peer OK: " << FLAGS_gpu;

    CUDA_CHECK(cudaMemcpyPeer(shared_gmem->mutable_gpu_data(), FLAGS_gpu,
        peer_gmem.gpu_data(), FLAGS_peer, shared_size));
    CUDA_CHECK(cudaStreamSynchronize(0));
    LOG(INFO) << "Peer copy data: finished; " << shared_gmem->mutable_gpu_data()
        << " @ " << shared_size << " <= " << peer_gmem.gpu_data();
  }

};

int train() {
  if (FLAGS_gpu < 0) {
    FLAGS_gpu = 0;
  }
  if (FLAGS_peer == -1 || FLAGS_peer == FLAGS_gpu) {
    FLAGS_peer = !FLAGS_gpu;
  }
  std::ostringstream info;
  info << "Use GPU with device ID: " << FLAGS_gpu << " + " << FLAGS_peer;
  
  Caffe::SetDevice(FLAGS_gpu);
  Caffe::set_mode(Caffe::GPU);
  SyncedMemory gmem(shared_size);
  CUDA_CHECK(cudaMemset(gmem.mutable_gpu_data(), 0x30, shared_size));
  const int **data = (const int **)gmem.mutable_cpu_data();
  info << "\n[" << data[0] << " " << data[1] << " "
      << data[2] << " " << data[3] << "]\n\n";
  LOG(INFO) << "\n" << info.str();
  
  shared_gmem = &gmem;
  CHECK((new ThreadPeer())->StartInternalThread()) << "Peer failed to start";
  sleep(3);
  
  std::ostringstream info2;
  info2 << "After sleep";
  info2 << "\n";
  data = (const int **)gmem.mutable_cpu_data();
  info2 << "\n[" << data[0] << " " << data[1] << " "
      << data[2] << " " << data[3] << "]\n\n";
  LOG(INFO) << "\n" << info2.str();
  return 0;
}
RegisterBrewFunction(train);



int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: peer <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/peer");
  }
}

