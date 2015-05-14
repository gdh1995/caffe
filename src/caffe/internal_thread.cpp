#include <boost/thread.hpp>
#include "caffe/internal_thread.hpp"
#include "caffe/util/mpi/interface.hpp"
#include <signal.h>

namespace caffe {

InternalThread::~InternalThread() {
  WaitForInternalThreadToExit();
}

bool InternalThread::is_started() const {
  return thread_.get() != NULL && thread_->joinable();
}

void InternalThread::EntryWrapper() {
  {
    sigset_t wait_set;
    sigemptyset(&wait_set);
    sigaddset(&wait_set, mpi::SIGSYNC);
    sigprocmask(SIG_BLOCK, &wait_set, NULL);
  }
  this->InternalThreadEntry();
}

bool InternalThread::StartInternalThread() {
  if (!WaitForInternalThreadToExit()) {
    return false;
  }
  try {
    thread_.reset(
        new boost::thread(&InternalThread::EntryWrapper, this));
  } catch (...) {
    return false;
  }
  return true;
}

/** Will not return until the internal thread has exited. */
bool InternalThread::WaitForInternalThreadToExit() {
  if (is_started()) {
    try {
      thread_->join();
    } catch (...) {
      return false;
    }
  }
  return true;
}

}  // namespace caffe
