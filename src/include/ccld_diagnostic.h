#ifndef RCCL_CCLD_DIAGNOSTIC_H_
#define RCCL_CCLD_DIAGNOSTIC_H_

#include <stdint.h>
#include <stdbool.h>

struct ncclInfo;
struct ncclComm;
struct ncclCcldDiag;
#ifdef ENABLE_COLLTRACE
struct ncclCollTrace;
#endif

// Record the host-side beginning of a collective operation.  The recorder is
// disabled by default and is intentionally independent of the transport path.
void rcclCcldRecordEnqueue(const struct ncclInfo* info);
bool rcclCcldEnabled();

// Release per-communicator diagnostic state after RCCL has accepted destroy.
void rcclCcldForgetComm(const struct ncclComm* comm);
void rcclCcldRecordDeviceDiag(const struct ncclComm* comm,
                              const struct ncclCcldDiag* diag);
void rcclCcldRecordDeviceSample(const struct ncclComm* comm, uint64_t opCount,
                               uint16_t channel, uint64_t sendCount,
                               uint64_t recvCount, uint64_t timestampNs,
                               uint64_t sendDelta, uint64_t recvDelta,
                               uint64_t elapsedNs, bool counterReset);

#ifdef ENABLE_COLLTRACE
void rcclCcldRecordDevice(const struct ncclComm* comm,
                          const struct ncclCollTrace* trace,
                          uint64_t send_count, uint64_t recv_count,
                          bool complete);
#endif

#endif
