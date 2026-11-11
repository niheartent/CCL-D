#include <mpi.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CHECK_MPI(x) do { int e=(x); if(e!=MPI_SUCCESS){ fprintf(stderr,"MPI error %d\n",e); MPI_Abort(MPI_COMM_WORLD,e); } } while(0)
#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){ fprintf(stderr,"HIP error %s\n",hipGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,(int)e); } } while(0)
#define CHECK_RCCL(x) do { ncclResult_t e=(x); if(e!=ncclSuccess){ fprintf(stderr,"RCCL error %s\n",ncclGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,(int)e); } } while(0)

static void sync(MPI_Comm world, ncclComm_t comm, hipStream_t stream) {
  CHECK_HIP(hipStreamSynchronize(stream));
  ncclResult_t async_error = ncclSuccess;
  CHECK_RCCL(ncclCommGetAsyncError(comm, &async_error));
  if (async_error != ncclSuccess) { fprintf(stderr, "RCCL async error %s\n", ncclGetErrorString(async_error)); MPI_Abort(world, (int)async_error); }
  CHECK_MPI(MPI_Barrier(world));
}

static bool check_close(const std::vector<float>& got, float expected, const char* name, int rank, int iter) {
  for (size_t i=0; i<got.size(); ++i) if (got[i] != expected) {
    fprintf(stderr, "%s FAIL rank=%d iter=%d index=%zu got=%g expected=%g\n", name, rank, iter, i, got[i], expected);
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  MPI_Comm world = MPI_COMM_WORLD;
  int rank=0, nranks=0; CHECK_MPI(MPI_Comm_rank(world,&rank)); CHECK_MPI(MPI_Comm_size(world,&nranks));
  int iters = argc > 1 ? std::atoi(argv[1]) : 20;
  size_t count = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 4096;
  int ndev=0; CHECK_HIP(hipGetDeviceCount(&ndev)); if(ndev<=0){fprintf(stderr,"no GPU\n"); MPI_Abort(world,1);} CHECK_HIP(hipSetDevice(rank % ndev));

  ncclUniqueId id; if(rank==0) CHECK_RCCL(ncclGetUniqueId(&id)); CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, world));
  ncclComm_t comm; CHECK_RCCL(ncclCommInitRank(&comm,nranks,id,rank));
  hipStream_t stream; CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  float *a=nullptr,*b=nullptr,*c=nullptr; CHECK_HIP(hipMalloc(&a, count*nranks*sizeof(float))); CHECK_HIP(hipMalloc(&b, count*nranks*sizeof(float))); CHECK_HIP(hipMalloc(&c, count*sizeof(float)));
  std::vector<float> host(count*nranks), recv(count*nranks), out(count);
  int failures=0;

  for(int iter=0; iter<iters; ++iter) {
    // AllGather: every rank contributes its rank+1 segment.
    std::fill(host.begin(), host.end(), float(rank+1)); CHECK_HIP(hipMemcpyAsync(a,host.data(),host.size()*sizeof(float),hipMemcpyHostToDevice,stream));
    CHECK_RCCL(ncclAllGather(a,b,count,ncclFloat,comm,stream)); sync(world,comm,stream);
    CHECK_HIP(hipMemcpy(recv.data(),b,recv.size()*sizeof(float),hipMemcpyDeviceToHost));
    for(int peer=0; peer<nranks; ++peer) for(size_t i=0;i<count;++i) if(recv[peer*count+i] != float(peer+1)) failures++;

    // ReduceScatter: each rank contributes rank+1; sum is deterministic.
    std::fill(host.begin(), host.end(), float(rank+1)); CHECK_HIP(hipMemcpyAsync(a,host.data(),host.size()*sizeof(float),hipMemcpyHostToDevice,stream));
    CHECK_RCCL(ncclReduceScatter(a,c,count,ncclFloat,ncclSum,comm,stream)); sync(world,comm,stream);
    CHECK_HIP(hipMemcpy(out.data(),c,out.size()*sizeof(float),hipMemcpyDeviceToHost));
    if(!check_close(out, float(nranks*(nranks+1)/2), "ReduceScatter", rank, iter)) failures++;

    // Broadcast: root owns 7, all ranks must receive 7.
    std::fill(host.begin(), host.end(), rank==0 ? 7.0f : 0.0f); CHECK_HIP(hipMemcpyAsync(a,host.data(),count*sizeof(float),hipMemcpyHostToDevice,stream));
    CHECK_RCCL(ncclBroadcast(a,b,count,ncclFloat,0,comm,stream)); sync(world,comm,stream);
    CHECK_HIP(hipMemcpy(out.data(),b,out.size()*sizeof(float),hipMemcpyDeviceToHost));
    if(!check_close(out, 7.0f, "Broadcast", rank, iter)) failures++;

    // AllToAll: sender rank -> receiver peer is encoded as rank*100+peer.
    for(int peer=0; peer<nranks; ++peer) for(size_t i=0;i<count;++i) host[peer*count+i] = float(rank*100+peer);
    CHECK_HIP(hipMemcpyAsync(a,host.data(),host.size()*sizeof(float),hipMemcpyHostToDevice,stream));
    CHECK_RCCL(ncclAllToAll(a,b,count,ncclFloat,comm,stream)); sync(world,comm,stream);
    CHECK_HIP(hipMemcpy(recv.data(),b,recv.size()*sizeof(float),hipMemcpyDeviceToHost));
    for(int sender=0; sender<nranks; ++sender) for(size_t i=0;i<count;++i) if(recv[sender*count+i] != float(sender*100+rank)) failures++;

    // P2P: ring send/recv has explicit peer semantics.
    int send_peer=(rank+1)%nranks, recv_peer=(rank-1+nranks)%nranks;
    std::fill(host.begin(), host.begin()+count, float(rank+1)); CHECK_HIP(hipMemcpyAsync(a,host.data(),count*sizeof(float),hipMemcpyHostToDevice,stream));
    CHECK_RCCL(ncclGroupStart()); CHECK_RCCL(ncclSend(a,count,ncclFloat,send_peer,comm,stream)); CHECK_RCCL(ncclRecv(c,count,ncclFloat,recv_peer,comm,stream)); CHECK_RCCL(ncclGroupEnd()); sync(world,comm,stream);
    CHECK_HIP(hipMemcpy(out.data(),c,out.size()*sizeof(float),hipMemcpyDeviceToHost));
    if(!check_close(out, float(recv_peer+1), "P2P", rank, iter)) failures++;
  }
  int global_failures=0; CHECK_MPI(MPI_Allreduce(&failures,&global_failures,1,MPI_INT,MPI_SUM,world));
  if(rank==0) { if(global_failures) printf("slot_regression FAIL ranks=%d iters=%d count=%zu mismatches=%d\n",nranks,iters,count,global_failures); else printf("slot_regression PASS ranks=%d iters=%d count=%zu checks=AllGather,ReduceScatter,Broadcast,AllToAll,P2P\n",nranks,iters,count); }
  CHECK_HIP(hipFree(a)); CHECK_HIP(hipFree(b)); CHECK_HIP(hipFree(c)); CHECK_HIP(hipStreamDestroy(stream)); CHECK_RCCL(ncclCommDestroy(comm)); MPI_Finalize(); return global_failures ? 2 : 0;
}
