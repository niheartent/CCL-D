#include <mpi.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#define CHECK_MPI(call) do { int e = (call); if (e != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, e); } while (0)
#define CHECK_HIP(call) do { hipError_t e = (call); if (e != hipSuccess) { std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CHECK_RCCL(call) do { ncclResult_t e = (call); if (e != ncclSuccess) { std::fprintf(stderr, "RCCL error: %s\n", ncclGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static long readVmPinKb() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmPin:") {
      long value = 0;
      std::string unit;
      status >> value >> unit;
      return value;
    }
    std::string rest;
    std::getline(status, rest);
  }
  return -1;
}

int main(int argc, char** argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  int rank = 0, nranks = 0, ndev = 0;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  const int groupWorks = argc > 1 ? std::atoi(argv[1]) : 256;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 2;
  const size_t count = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 4096;
  if (groupWorks < 1 || iterations < 1 || count < 1) MPI_Abort(MPI_COMM_WORLD, 2);
  CHECK_HIP(hipGetDeviceCount(&ndev));
  if (ndev < 1 || nranks < 2) MPI_Abort(MPI_COMM_WORLD, 3);
  CHECK_HIP(hipSetDevice(rank % ndev));
  const long vmPinBeforeKb = readVmPinKb();

  ncclUniqueId id;
  if (rank == 0) CHECK_RCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm = nullptr;
  CHECK_RCCL(ncclCommInitRank(&comm, nranks, id, rank));
  const long vmPinAfterInitKb = readVmPinKb();

  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipEvent_t eventStart, eventStop;
  CHECK_HIP(hipEventCreate(&eventStart));
  CHECK_HIP(hipEventCreate(&eventStop));
  float* grouped = nullptr;
  float* p2pSendRecv = nullptr;
  CHECK_HIP(hipMalloc(&grouped, groupWorks * count * sizeof(float)));
  CHECK_HIP(hipMalloc(&p2pSendRecv, 2 * count * sizeof(float)));
  std::vector<float> host(groupWorks * count, static_cast<float>(rank + 1));
  std::vector<float> p2pHost(count, static_cast<float>(rank + 1));
  const float expected = static_cast<float>(nranks * (nranks + 1) / 2);
  const int sendPeer = (rank + 1) % nranks;
  const int recvPeer = (rank + nranks - 1) % nranks;
  int errors = 0;
  std::vector<float> groupGpuMs;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    // AllReduce writes the result back into the host staging buffer below.
    // Restore each input before the next iteration so every round reduces the
    // same rank-local value instead of reducing the previous round's result.
    std::fill(host.begin(), host.end(), static_cast<float>(rank + 1));
    std::fill(p2pHost.begin(), p2pHost.end(), static_cast<float>(rank + 1));
    CHECK_HIP(hipMemcpyAsync(grouped, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipMemcpyAsync(p2pSendRecv, p2pHost.data(), count * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipEventRecord(eventStart, stream));
    CHECK_RCCL(ncclGroupStart());
    for (int work = 0; work < groupWorks; ++work) {
      float* buffer = grouped + static_cast<size_t>(work) * count;
      CHECK_RCCL(ncclAllReduce(buffer, buffer, count, ncclFloat32, ncclSum, comm, stream));
    }
    // Mix P2P work into the same group as the collective work.
    CHECK_RCCL(ncclSend(p2pSendRecv, count, ncclFloat32, sendPeer, comm, stream));
    CHECK_RCCL(ncclRecv(p2pSendRecv + count, count, ncclFloat32, recvPeer, comm, stream));
    CHECK_RCCL(ncclGroupEnd());
    CHECK_HIP(hipEventRecord(eventStop, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    if (iteration > 0) {
      float elapsedMs = 0.0f;
      CHECK_HIP(hipEventElapsedTime(&elapsedMs, eventStart, eventStop));
      groupGpuMs.push_back(elapsedMs);
    }
    CHECK_HIP(hipMemcpy(host.data(), grouped, host.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(p2pHost.data(), p2pSendRecv + count, count * sizeof(float), hipMemcpyDeviceToHost));
    for (float value : host) if (value != expected) { ++errors; break; }
    for (float value : p2pHost) if (value != static_cast<float>(recvPeer + 1)) { ++errors; break; }
  }

  int totalErrors = 0;
  CHECK_MPI(MPI_Allreduce(&errors, &totalErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  if (rank == 0) {
    std::printf("grouped_work %s ranks=%d group_works=%d iterations=%d count=%zu mixed_p2p=1 errors=%d\n",
                totalErrors == 0 ? "PASS" : "FAIL", nranks, groupWorks, iterations, count, totalErrors);
    if (!groupGpuMs.empty()) {
      std::sort(groupGpuMs.begin(), groupGpuMs.end());
      const double mean = std::accumulate(groupGpuMs.begin(), groupGpuMs.end(), 0.0) / groupGpuMs.size();
      const size_t p50 = (groupGpuMs.size() - 1) / 2;
      const size_t p95 = (groupGpuMs.size() * 95 - 1) / 100;
      std::printf("group_gpu_event_ms samples=%zu mean=%.6f p50=%.6f p95=%.6f\n",
                  groupGpuMs.size(), mean, groupGpuMs[p50], groupGpuMs[p95]);
    }
  }
  std::printf("resource_probe rank=%d vm_pin_before_kb=%ld vm_pin_after_init_kb=%ld vm_pin_delta_kb=%ld\n",
              rank, vmPinBeforeKb, vmPinAfterInitKb,
              vmPinBeforeKb >= 0 && vmPinAfterInitKb >= 0 ? vmPinAfterInitKb - vmPinBeforeKb : -1);
  CHECK_HIP(hipFree(grouped));
  CHECK_HIP(hipFree(p2pSendRecv));
  CHECK_HIP(hipStreamDestroy(stream));
  CHECK_HIP(hipEventDestroy(eventStart));
  CHECK_HIP(hipEventDestroy(eventStop));
  CHECK_RCCL(ncclCommDestroy(comm));
  CHECK_MPI(MPI_Finalize());
  return totalErrors == 0 ? 0 : 4;
}
