#include <mpi.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, e); } while (0)
#define HIP_CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define NCCL_CHECK(call) do { ncclResult_t e = (call); if (e != ncclSuccess) { std::fprintf(stderr, "RCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank = 0, nranks = 0, local_rank = 0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  const char* local = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (local) local_rank = std::atoi(local);
  if (nranks < 2) {
    if (rank == 0) std::fprintf(stderr, "need at least two MPI ranks\n");
    MPI_Finalize();
    return 2;
  }
  HIP_CHECK(hipSetDevice(local_rank));
  ncclUniqueId id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr int count = 1024;
  std::vector<float> host(count, static_cast<float>(rank + 1));
  float* device = nullptr;
  HIP_CHECK(hipMalloc(&device, count * sizeof(float)));
  HIP_CHECK(hipMemcpyAsync(device, host.data(), count * sizeof(float), hipMemcpyHostToDevice, stream));
  NCCL_CHECK(ncclAllReduce(device, device, count, ncclFloat32, ncclSum, comm, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost));
  const float expected = static_cast<float>(nranks * (nranks + 1) / 2);
  int bad = 0;
  for (float v : host) if (v != expected) { bad = 1; break; }
  std::printf("rank=%d host=%s local_rank=%d result=%s value=%g\n", rank, std::getenv("HOSTNAME") ?: "?", local_rank, bad ? "FAIL" : "PASS", host[0]);
  MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
  NCCL_CHECK(ncclCommDestroy(comm));
  HIP_CHECK(hipFree(device));
  HIP_CHECK(hipStreamDestroy(stream));
  MPI_Finalize();
  return bad ? 1 : 0;
}
