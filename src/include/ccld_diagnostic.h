#ifndef RCCL_CCLD_DIAGNOSTIC_H_
#define RCCL_CCLD_DIAGNOSTIC_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

struct ncclInfo;
struct ncclComm;
struct ncclKernelPlan;
#ifdef ENABLE_COLLTRACE
struct ncclCollTrace;
#endif

void rcclCcldRecordEnqueue(struct ncclInfo* info);
void rcclCcldRegisterBody(const struct ncclComm* comm, uint64_t counter,
                          uint16_t channel,
                          int algorithm, int protocol);
void rcclCcldFinalizeDiagnostics(const struct ncclComm* comm,
                                 const uint64_t* counters, size_t count);
void rcclCcldDrainDiagnostics(const struct ncclComm* comm);
void rcclCcldSampleDiagnostics(const struct ncclComm* comm, uint64_t timestampNs,
                               uint64_t sampleIntervalNs);
void rcclCcldRegisterPlanCounter(struct ncclKernelPlan* plan, uint64_t counter);
bool rcclCcldEnabled();
void rcclCcldForgetComm(const struct ncclComm* comm);
void rcclCcldRecordDeviceSample(const struct ncclComm* comm, uint64_t opCount,
                               uint16_t channel, uint64_t sendCount,
                               uint64_t recvCount, uint64_t timestampNs,
                               uint64_t sendDelta, uint64_t recvDelta,
                               uint64_t elapsedNs, bool counterReset);
void rcclCcldRecordResource(const struct ncclComm* comm, size_t frameBytes,
                            size_t blockBytes, int slots);
void rcclCcldRecordThreadResource(const struct ncclComm* comm,
                                  uint64_t threadCpuNs, uint64_t threadWallNs,
                                  uint64_t sampleCpuNs, uint64_t sampleWallNs,
                                  uint64_t sampleCalls, uint64_t deviceSamples,
                                  uint64_t sampleIntervalNs);
uint64_t rcclCcldSampleCount(const struct ncclComm* comm);
bool rcclCcldStartThread(struct ncclComm* comm, uint64_t sampleIntervalNs);
void rcclCcldNotifyCompletion(const struct ncclComm* comm);
void rcclCcldStopThread(struct ncclComm* comm);
#ifdef ENABLE_COLLTRACE
void rcclCcldRecordDevice(const struct ncclComm* comm,
                          const struct ncclCollTrace* trace,
                          uint64_t send_count, uint64_t recv_count,
                          bool complete);
#endif

#endif
