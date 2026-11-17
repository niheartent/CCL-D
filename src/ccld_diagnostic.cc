#include "ccld_diagnostic.h"
#include "collectives.h"
#include "comm.h"
#include "nccl.h"
#include "device.h"
#include "ccld_record_pool.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <memory>
#include <map>
#include <pthread.h>
#include <set>
#include <sstream>
#include <string>
#include <time.h>
#include <unordered_map>

namespace {
std::mutex g_lock;
std::ofstream g_log;
std::atomic<bool> g_initialized{false};
std::unordered_map<const ncclComm*, std::unique_ptr<rccl_ccld::RecordPool>> g_pools;
struct CompletedEvent { uint64_t counter; uint64_t finishNs; uint64_t finishWallNs; };
std::unordered_map<const ncclComm*, std::vector<CompletedEvent>> g_completed;
struct OperationInfo {
  uint64_t startNs = 0;
  uint64_t count = 0;
  uint64_t bytes = 0;
  int datatype = 0;
  int root = 0;
  int algorithm = -1;
  int protocol = -1;
  uint64_t opCount = 0;
  std::string op;
  std::set<uint16_t> channels;
  std::map<uint16_t, std::pair<uint64_t, uint64_t>> finalCounts;
  // Number of launched kernel plans that carry this TraceID.  A collective
  // may be split into several plans; the frame remains live until all of them
  // have delivered their completion callback.
  size_t pendingPlans = 0;
};
struct SamplePrevious { uint64_t timestamp, send, recv; };
std::unordered_map<const ncclComm*, std::unordered_map<uint64_t, OperationInfo>> g_operations;
std::unordered_map<std::string, SamplePrevious> g_samplePrevious;
std::unordered_map<const ncclComm*, uint64_t> g_sampleCounts;

struct DiagnosticThreadState {
  const ncclComm* comm = nullptr;
  pthread_t thread{};
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
  bool stop = false;
  uint64_t notificationGeneration = 0;
  uint64_t sampleIntervalNs = 0;
};
std::unordered_map<const ncclComm*, std::unique_ptr<DiagnosticThreadState>> g_threads;

bool enabled() {
  return rcclParamCcldEnable() != 0 && rcclParamCcldDiagMode() != 0;
}

void replaceAll(std::string& value, const std::string& token, const std::string& replacement) {
  size_t pos = 0;
  while ((pos = value.find(token, pos)) != std::string::npos) {
    value.replace(pos, token.size(), replacement);
    pos += replacement.size();
  }
}

std::string rankPath(const char* configured, int rank) {
  std::string path = configured && *configured ? configured : "/tmp/rccl-ccld.jsonl";
  const std::string rankText = std::to_string(rank);
  const bool templated = path.find("{rank}") != std::string::npos || path.find("${RANK}") != std::string::npos || path.find("%r") != std::string::npos;
  replaceAll(path, "{rank}", rankText); replaceAll(path, "${RANK}", rankText); replaceAll(path, "%r", rankText);
  if (!templated) {
    const size_t slash = path.find_last_of('/'), dot = path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) path.insert(dot, ".rank" + rankText);
    else path += ".rank" + rankText;
  }
  return path;
}

void init(const ncclComm* comm) {
  if (g_initialized.exchange(true)) return;
  const char* configured = std::getenv("RCCL_CCLD_LOG");
  g_log.open(rankPath(configured, comm ? comm->rank : -1), std::ios::out | std::ios::app);
}

uint64_t nowNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
uint64_t wallNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}
const char* safe(const char* value) { return value ? value : ""; }

uint64_t elapsedTimespecNs(const struct timespec& begin, const struct timespec& end) {
  int64_t sec = static_cast<int64_t>(end.tv_sec) - static_cast<int64_t>(begin.tv_sec);
  int64_t ns = static_cast<int64_t>(end.tv_nsec) - static_cast<int64_t>(begin.tv_nsec);
  if (ns < 0) { --sec; ns += 1000000000LL; }
  return sec > 0 ? static_cast<uint64_t>(sec) * 1000000000ULL + static_cast<uint64_t>(ns)
                 : static_cast<uint64_t>(ns);
}

double algorithmBandwidthFactor(const std::string& op, int nranks) {
  if (nranks <= 0) return 0.0;
  const double peerFactor = static_cast<double>(nranks - 1) / nranks;
  if (op == "AllReduce" || op == "Reduce") return 2.0 * peerFactor;
  if (op == "AllGather" || op == "ReduceScatter" ||
      op == "Broadcast" || op == "AllToAll") return peerFactor;
  // P2P has one sender/receiver edge; MIXED is a grouped P2P parent.
  if (op == "Send" || op == "Recv" || op == "P2P" || op == "MIXED") return 1.0;
  return 0.0;
}

void* diagnosticThreadMain(void* arg) {
  DiagnosticThreadState* state = static_cast<DiagnosticThreadState*>(arg);
  const ncclComm* comm = state->comm;
  struct timespec cpuStart{}, wallStart{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuStart);
  clock_gettime(CLOCK_MONOTONIC, &wallStart);
  uint64_t sampleCpuNs = 0, sampleWallNs = 0, sampleCalls = 0;
  const uint64_t interval = state->sampleIntervalNs;
  while (true) {
    pthread_mutex_lock(&state->mutex);
    const uint64_t observedGeneration = state->notificationGeneration;
    const bool stopping = state->stop;
    pthread_mutex_unlock(&state->mutex);
    if (stopping) break;
    struct timespec sampleCpuStart{}, sampleWallStart{}, sampleCpuEnd{}, sampleWallEnd{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &sampleCpuStart);
    clock_gettime(CLOCK_MONOTONIC, &sampleWallStart);
    rcclCcldDrainDiagnostics(comm);
    rcclCcldSampleDiagnostics(comm, nowNs(), interval);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &sampleCpuEnd);
    clock_gettime(CLOCK_MONOTONIC, &sampleWallEnd);
    sampleCpuNs += elapsedTimespecNs(sampleCpuStart, sampleCpuEnd);
    sampleWallNs += elapsedTimespecNs(sampleWallStart, sampleWallEnd);
    ++sampleCalls;

    pthread_mutex_lock(&state->mutex);
    if (state->stop) { pthread_mutex_unlock(&state->mutex); break; }
    if (state->notificationGeneration != observedGeneration) {
      pthread_mutex_unlock(&state->mutex);
      continue;
    }
    struct timespec deadline{};
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += static_cast<time_t>(interval / 1000000000ULL);
    deadline.tv_nsec += static_cast<long>(interval % 1000000000ULL);
    if (deadline.tv_nsec >= 1000000000L) { ++deadline.tv_sec; deadline.tv_nsec -= 1000000000L; }
    pthread_cond_timedwait(&state->condition, &state->mutex, &deadline);
    const bool stop = state->stop;
    pthread_mutex_unlock(&state->mutex);
    if (stop) break;
  }
  rcclCcldDrainDiagnostics(comm);
  struct timespec cpuEnd{}, wallEnd{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuEnd);
  clock_gettime(CLOCK_MONOTONIC, &wallEnd);
  rcclCcldRecordThreadResource(comm, elapsedTimespecNs(cpuStart, cpuEnd),
                               elapsedTimespecNs(wallStart, wallEnd),
                               sampleCpuNs, sampleWallNs, sampleCalls,
                               rcclCcldSampleCount(comm), interval);
  return nullptr;
}
}

bool rcclCcldEnabled() { return enabled(); }

bool rcclCcldStartThread(struct ncclComm* comm, uint64_t sampleIntervalNs) {
  if (!comm || !enabled()) return false;
  std::lock_guard<std::mutex> guard(g_lock);
  if (g_threads.find(comm) != g_threads.end()) return true;
  auto state = std::make_unique<DiagnosticThreadState>();
  state->comm = comm;
  state->sampleIntervalNs = sampleIntervalNs ? sampleIntervalNs :
      static_cast<uint64_t>(rcclParamCcldSampleUs()) * 1000ULL;
  DiagnosticThreadState* raw = state.get();
  g_threads.emplace(comm, std::move(state));
  if (pthread_create(&raw->thread, nullptr, diagnosticThreadMain, raw) != 0) {
    g_threads.erase(comm);
    return false;
  }
  return true;
}

void rcclCcldNotifyCompletion(const struct ncclComm* comm) {
  std::lock_guard<std::mutex> guard(g_lock);
  auto it = g_threads.find(comm);
  if (it == g_threads.end()) return;
  pthread_mutex_lock(&it->second->mutex);
  ++it->second->notificationGeneration;
  pthread_cond_signal(&it->second->condition);
  pthread_mutex_unlock(&it->second->mutex);
}

void rcclCcldStopThread(struct ncclComm* comm) {
  if (!comm) return;
  std::unique_ptr<DiagnosticThreadState> state;
  {
    std::lock_guard<std::mutex> guard(g_lock);
    auto it = g_threads.find(comm);
    if (it == g_threads.end()) return;
    state = std::move(it->second);
    g_threads.erase(it);
  }
  pthread_mutex_lock(&state->mutex);
  state->stop = true;
  pthread_cond_signal(&state->condition);
  pthread_mutex_unlock(&state->mutex);
  pthread_join(state->thread, nullptr);
  pthread_cond_destroy(&state->condition);
  pthread_mutex_destroy(&state->mutex);
}

void rcclCcldRecordEnqueue(struct ncclInfo* info) {
  if (!info || !info->comm || !enabled()) return;
  info->ccldCounter = UINT64_MAX;
  info->ccldKernelIndex = UINT32_MAX;
  if (info->comm->nRanks <= 1 || info->count == 0 || info->comm->ccldDiagMode == 0) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(info->comm);
  if (!g_log || !info->comm->ccldDiag || info->comm->ccldDiagSlots <= 0) return;
  const ncclComm* comm = info->comm;
  auto& pool = g_pools[comm];
  if (!pool) {
    const size_t blocks = comm->ccldDiagSlots > 0 ? static_cast<size_t>(comm->ccldDiagSlots) : 1u;
    pool.reset(new rccl_ccld::RecordPool(blocks));
  }
  // The identity is the communicator-wide collective operation counter.
  // The record pool only selects storage; it never creates TraceID values.
  const uint64_t counter = comm->opCount;
  uint32_t blockIndex = UINT32_MAX;
  info->ccldCounter = counter;
  const uint64_t enqueueNs = nowNs();
  const uint64_t enqueueWallNs = wallNs();
  auto existing = g_operations[comm].find(counter);
  if (existing != g_operations[comm].end()) {
    if (!pool->find(counter, &blockIndex)) {
      info->ccldCounter = UINT64_MAX;
      info->ccldKernelIndex = UINT32_MAX;
      return;
    }
    // Grouped P2P calls share one parent operation identity. Keep individual
    // enqueue rows in the log, while the lifecycle record aggregates metadata.
    OperationInfo& operation = existing->second;
    const std::string currentOp = safe(info->opName);
    const bool previousP2p = operation.op == "Send" || operation.op == "Recv" || operation.op == "P2P";
    const bool currentP2p = currentOp == "Send" || currentOp == "Recv";
    if (operation.op != currentOp) operation.op = previousP2p && currentP2p ? "P2P" : "MIXED";
    operation.count += info->count;
    operation.bytes += info->count * ncclTypeSize(info->datatype);
    if (operation.root != info->root) operation.root = -1;
    if (operation.datatype != static_cast<int>(info->datatype)) operation.datatype = -1;
    info->ccldKernelIndex = blockIndex;
    g_log << "{\"schema\":1,\"phase\":\"enqueue\",\"timestamp_ns\":" << enqueueNs
          << ",\"timestamp_wall_ns\":" << enqueueWallNs
          << ",\"trace_id\":\"" << std::hex << comm->commHash << ':' << std::dec << counter
          << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
          << "\",\"counter\":" << counter << ",\"op_count\":" << comm->opCount
          << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
          << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
          << ",\"local_rank\":" << comm->localRank << ",\"local_ranks\":" << comm->localRanks
          << ",\"op\":\"" << currentOp << "\",\"count\":" << info->count
          << ",\"bytes\":" << info->count * ncclTypeSize(info->datatype)
          << ",\"datatype\":" << static_cast<int>(info->datatype)
          << ",\"root\":" << info->root << ",\"host_status\":\"entered\"}\n";
    g_log.flush();
    return;
  }
  if (!pool->reserve(counter, &blockIndex)) {
    g_log << "{\"schema\":1,\"phase\":\"enqueue\",\"timestamp_ns\":" << enqueueNs
          << ",\"timestamp_wall_ns\":" << enqueueWallNs
          << ",\"trace_id\":\"" << std::hex << comm->commHash << ':' << std::dec << counter
          << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
          << "\",\"counter\":" << counter << ",\"op_count\":" << comm->opCount
          << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
          << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
          << ",\"local_rank\":" << comm->localRank << ",\"local_ranks\":" << comm->localRanks
          << ",\"op\":\"" << safe(info->opName) << "\",\"count\":" << info->count
          << ",\"bytes\":" << info->count * ncclTypeSize(info->datatype)
          << ",\"datatype\":" << static_cast<int>(info->datatype)
          << ",\"root\":" << info->root << ",\"host_status\":\"entered\",\"diagnostic_status\":\"OBSERVATION_DROPPED\"}\n";
    g_log.flush();
    return;
  }
  auto& operation = g_operations[comm][counter];
  operation.count = info->count;
  operation.startNs = enqueueNs;
  operation.bytes = info->count * ncclTypeSize(info->datatype);
  operation.datatype = static_cast<int>(info->datatype);
  operation.root = info->root;
  operation.opCount = comm->opCount;
  operation.op = safe(info->opName);

  struct ncclCcldDiag* block = comm->ccldDiag + blockIndex;
  // Change the generation first so an older in-flight kernel stops recording
  // before the body is cleared and reused.
  __atomic_store_n(&block->header.mode, 0ULL, __ATOMIC_RELEASE);
  __atomic_store_n(&block->header.counter, counter, __ATOMIC_RELEASE);
  for (int channel = 0; channel < MAXCHANNELS; ++channel) {
    __atomic_store_n(&block->channels[channel].sendCount, 0ULL, __ATOMIC_RELAXED);
    __atomic_store_n(&block->channels[channel].recvCount, 0ULL, __ATOMIC_RELAXED);
  }
  block->header.kernelIndex = blockIndex;
  block->header.detectChannel = 0;
  block->header.traceId = rccl_ccld::makeTraceId(comm->commHash, counter);
  __atomic_store_n(&block->header.mode, static_cast<uint64_t>(comm->ccldDiagMode), __ATOMIC_RELEASE);
  info->ccldCounter = counter;
  info->ccldKernelIndex = blockIndex;

  g_log << "{\"schema\":1,\"phase\":\"enqueue\",\"timestamp_ns\":" << operation.startNs
        << ",\"timestamp_wall_ns\":" << enqueueWallNs
        << ",\"trace_id\":\"" << std::hex << comm->commHash << ':' << std::dec << counter
        << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"counter\":" << counter << ",\"op_count\":" << comm->opCount
        << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
        << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
        << ",\"local_rank\":" << comm->localRank << ",\"local_ranks\":" << comm->localRanks
        << ",\"op\":\"" << operation.op << "\",\"count\":" << operation.count
        << ",\"bytes\":" << operation.bytes << ",\"datatype\":" << operation.datatype
        << ",\"root\":" << operation.root << ",\"host_status\":\"entered\"}\n";
  g_log.flush();
}

void rcclCcldRegisterBody(const struct ncclComm* comm, uint64_t counter,
                          uint16_t channel, int algorithm, int protocol) {
  if (!comm || !enabled() || !comm->ccldDiag || channel >= MAXCHANNELS) return;
  std::lock_guard<std::mutex> guard(g_lock);
  auto opIt = g_operations.find(comm);
  if (opIt == g_operations.end()) return;
  auto it = opIt->second.find(counter);
  if (it == opIt->second.end()) return;
  it->second.channels.insert(channel);
  if (algorithm >= 0) it->second.algorithm = algorithm;
  if (protocol >= 0) it->second.protocol = protocol;
  uint32_t index = UINT32_MAX;
  auto pool = g_pools.find(comm);
  if (pool != g_pools.end() && pool->second->find(counter, &index))
    comm->ccldDiag[index].header.detectChannel = it->second.channels.size();
}

void rcclCcldRegisterPlanCounter(struct ncclKernelPlan* plan, uint64_t counter) {
  if (!plan || counter == UINT64_MAX || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  for (int i = 0; i < plan->ccldCounterCount; ++i) if (plan->ccldCounters[i] == counter) return;
  if (plan->ccldCounterCount == plan->ccldCounterCapacity) {
    const int64_t configuredInitial = rcclParamCcldPlanCounterInitial();
    const int initial = configuredInitial > 0 && configuredInitial <= INT32_MAX
        ? static_cast<int>(configuredInitial) : 1;
    const int next = plan->ccldCounterCapacity ? plan->ccldCounterCapacity * 2 : initial;
    uint64_t* p = static_cast<uint64_t*>(std::realloc(plan->ccldCounters, sizeof(uint64_t) * next));
    if (!p) return;
    plan->ccldCounters = p; plan->ccldCounterCapacity = next;
  }
  plan->ccldCounters[plan->ccldCounterCount++] = counter;
  auto commIt = g_operations.find(plan->comm);
  if (commIt != g_operations.end()) {
    auto opIt = commIt->second.find(counter);
    if (opIt != commIt->second.end()) ++opIt->second.pendingPlans;
  }
}

static void writeCompletedDiagnostics(const struct ncclComm* comm, const CompletedEvent* events, size_t count) {
  if (!comm || !enabled() || !events) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(comm);
  auto ops = g_operations.find(comm);
  if (ops == g_operations.end()) return;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t counter = events[i].counter;
    auto opIt = ops->second.find(counter);
    if (opIt == ops->second.end()) continue;
    const uint64_t finishNs = events[i].finishNs;
    const uint64_t durationNs = finishNs > opIt->second.startNs ? finishNs - opIt->second.startNs : 0;
    const long double algorithmBytes = static_cast<long double>(opIt->second.bytes) *
        algorithmBandwidthFactor(opIt->second.op, comm->nRanks);
    const uint64_t algorithmBandwidthBps = durationNs
        ? static_cast<uint64_t>(algorithmBytes * 1000000000.0L / durationNs) : 0;
    for (const auto& channelEntry : opIt->second.finalCounts) {
      const uint16_t channel = channelEntry.first;
      const uint64_t sendCount = channelEntry.second.first;
      const uint64_t recvCount = channelEntry.second.second;
    g_log << "{\"schema\":1,\"phase\":\"complete\",\"timestamp_ns\":" << finishNs
          << ",\"timestamp_wall_ns\":" << events[i].finishWallNs
            << ",\"trace_id\":\"" << std::hex << comm->commHash << ':' << std::dec << counter
            << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
            << "\",\"counter\":" << counter << ",\"op_count\":" << opIt->second.opCount
            << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
            << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
            << ",\"op\":\"" << opIt->second.op << "\",\"count\":" << opIt->second.count
            << ",\"bytes\":" << opIt->second.bytes << ",\"datatype\":" << opIt->second.datatype
            << ",\"root\":" << opIt->second.root << ",\"channel\":" << channel
            << ",\"algorithm\":" << opIt->second.algorithm
            << ",\"protocol\":" << opIt->second.protocol
            << ",\"send_count\":" << sendCount << ",\"recv_count\":" << recvCount
            << ",\"start_timestamp\":" << opIt->second.startNs
            << ",\"end_timestamp\":" << finishNs
            << ",\"duration_ns\":" << durationNs
            << ",\"algorithm_bandwidth_Bps\":" << algorithmBandwidthBps
            << ",\"algorithm_bandwidth_GBps\":" << (static_cast<long double>(algorithmBandwidthBps) / 1000000000.0L)
            << ",\"complete\":true";
      if (opIt->second.algorithm == NCCL_ALGO_TREE) {
        const ncclTree& tree = comm->channels[channel].tree;
        g_log << ",\"tree_up\":" << tree.up << ",\"tree_down0\":" << tree.down[0]
              << ",\"tree_down1\":" << tree.down[1] << ",\"tree_down2\":" << tree.down[2]
              << ",\"tree_depth\":" << tree.depth;
      } else if (opIt->second.algorithm == NCCL_ALGO_RING) {
        const ncclRing& ring = comm->channels[channel].ring;
        g_log << ",\"ring_prev\":" << ring.prev << ",\"ring_next\":" << ring.next;
      }
      g_log << "}\n";
    }
    g_log.flush();
    ops->second.erase(opIt);
  }
}

void rcclCcldFinalizeDiagnostics(const struct ncclComm* comm, const uint64_t* counters, size_t count) {
  if (!comm || !enabled() || !counters || count == 0) return;
  const uint64_t finish = nowNs();
  const uint64_t finishWall = wallNs();
  std::unique_lock<std::mutex> guard(g_lock);
  auto pool = g_pools.find(comm);
  auto ops = g_operations.find(comm);
  if (pool == g_pools.end() || ops == g_operations.end() || !comm->ccldDiag) return;
  std::vector<CompletedEvent>& pending = g_completed[comm];
  for (size_t i = 0; i < count; ++i) {
    uint32_t blockIndex = UINT32_MAX;
    if (!pool->second->find(counters[i], &blockIndex)) continue;
    auto opIt = ops->second.find(counters[i]);
    if (opIt == ops->second.end()) continue;
    struct ncclCcldDiag* block = comm->ccldDiag + blockIndex;
    if (block->header.counter != counters[i] || block->header.mode == 0) continue;
    for (uint16_t channel : opIt->second.channels) {
      const rccl_ccld::ChannelBody& body = block->channels[channel];
      opIt->second.finalCounts[channel] = std::make_pair(body.sendCount, body.recvCount);
    }
    if (opIt->second.pendingPlans > 0) --opIt->second.pendingPlans;
    // This callback represents one kernel. Keep the frame and accumulated
    // counts alive while another plan for the same TraceID is outstanding.
    if (opIt->second.pendingPlans != 0) continue;
    // The callback is ordered after all work that can update this body. Copy
    // the final counts before releasing the block; the polling thread only
    // needs the host snapshot to serialize the JSON record later.
    block->header.mode = 0;
    pool->second->release(counters[i]);
    const std::string prefix = std::to_string(reinterpret_cast<uintptr_t>(comm)) + ":" +
                               std::to_string(counters[i]) + ":";
    for (auto sample = g_samplePrevious.begin(); sample != g_samplePrevious.end();) {
      if (sample->first.compare(0, prefix.size(), prefix) == 0) sample = g_samplePrevious.erase(sample);
      else ++sample;
    }
    pending.push_back(CompletedEvent{counters[i], finish, finishWall});
  }
  guard.unlock();
  rcclCcldNotifyCompletion(comm);
}

void rcclCcldDrainDiagnostics(const struct ncclComm* comm) {
  if (!comm || !enabled()) return;
  std::vector<CompletedEvent> pending;
  {
    std::lock_guard<std::mutex> guard(g_lock);
    auto it = g_completed.find(comm);
    if (it == g_completed.end() || it->second.empty()) return;
    pending.swap(it->second);
  }
  writeCompletedDiagnostics(comm, pending.data(), pending.size());
}

void rcclCcldForgetComm(const struct ncclComm* comm) {
  if (!comm) return;
  std::lock_guard<std::mutex> guard(g_lock);
  g_pools.erase(comm); g_operations.erase(comm); g_completed.erase(comm); g_sampleCounts.erase(comm);
  const std::string prefix = std::to_string(reinterpret_cast<uintptr_t>(comm)) + ":";
  for (auto sample = g_samplePrevious.begin(); sample != g_samplePrevious.end();) {
    if (sample->first.compare(0, prefix.size(), prefix) == 0) sample = g_samplePrevious.erase(sample);
    else ++sample;
  }
}

void rcclCcldRecordResource(const struct ncclComm* comm, size_t frameBytes,
                            size_t blockBytes, int slots) {
  if (!comm || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(comm);
  if (!g_log) return;
  g_log << "{\"schema\":1,\"phase\":\"resource\",\"resource_status\":\"allocated\""
        << ",\"rank\":" << comm->rank << ",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"slots\":" << slots << ",\"frame_block_bytes\":" << blockBytes
        << ",\"frame_pinned_bytes\":" << frameBytes << "}\n";
  g_log.flush();
}

void rcclCcldRecordThreadResource(const struct ncclComm* comm,
                                  uint64_t threadCpuNs, uint64_t threadWallNs,
                                  uint64_t sampleCpuNs, uint64_t sampleWallNs,
                                  uint64_t sampleCalls, uint64_t deviceSamples,
                                  uint64_t sampleIntervalNs) {
  if (!comm || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(comm);
  if (!g_log) return;
  const double ratio = threadWallNs ? static_cast<double>(threadCpuNs) / static_cast<double>(threadWallNs) : 0.0;
  const double diagnosticRatio = threadWallNs ? static_cast<double>(sampleCpuNs) / static_cast<double>(threadWallNs) : 0.0;
  g_log << "{\"schema\":1,\"phase\":\"resource\",\"resource_status\":\"thread_complete\""
        << ",\"rank\":" << comm->rank << ",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"thread_cpu_ns\":" << threadCpuNs << ",\"thread_wall_ns\":" << threadWallNs
        << ",\"thread_cpu_ratio\":" << ratio
        << ",\"diagnostic_sample_cpu_ns\":" << sampleCpuNs
        << ",\"diagnostic_sample_wall_ns\":" << sampleWallNs
        << ",\"diagnostic_cpu_ratio\":" << diagnosticRatio
        << ",\"sample_calls\":" << sampleCalls
        << ",\"device_samples\":" << deviceSamples
        << ",\"sample_interval_ns\":" << sampleIntervalNs << "}\n";
  g_log.flush();
}

uint64_t rcclCcldSampleCount(const struct ncclComm* comm) {
  std::lock_guard<std::mutex> guard(g_lock);
  auto it = g_sampleCounts.find(comm);
  return it == g_sampleCounts.end() ? 0 : it->second;
}

void rcclCcldRecordDeviceSample(const struct ncclComm* comm, uint64_t counter, uint16_t channel,
                               uint64_t sendCount, uint64_t recvCount, uint64_t timestampNs,
                               uint64_t sendDelta, uint64_t recvDelta, uint64_t elapsedNs,
                               bool counterReset) {
  if (!comm || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock); init(comm); if (!g_log) return;
  g_log << "{\"schema\":1,\"phase\":\"sample\",\"timestamp_ns\":" << timestampNs
        << ",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"counter\":" << counter << ",\"rank\":" << comm->rank
        << ",\"nranks\":" << comm->nRanks << ",\"node\":" << comm->node
        << ",\"local_rank\":" << comm->localRank << ",\"local_ranks\":" << comm->localRanks
        << ",\"channel\":" << channel
        << ",\"send_count\":" << sendCount << ",\"recv_count\":" << recvCount
        << ",\"send_delta\":" << sendDelta << ",\"recv_delta\":" << recvDelta
        << ",\"elapsed_ns\":" << elapsedNs << ",\"counter_reset\":" << (counterReset ? "true" : "false")
        // AntCCL defines sendRate/recvRate from count changes in one sampling
        // interval. Keep the raw interval deltas as the primary metrics and
        // expose a frequency only as a derived, explicitly named field.
        << ",\"send_rate\":" << sendDelta
        << ",\"recv_rate\":" << recvDelta
        << ",\"send_frequency_hz\":" << (elapsedNs ? static_cast<uint64_t>(static_cast<long double>(sendDelta) * 1000000000.0L / elapsedNs) : 0)
        << ",\"recv_frequency_hz\":" << (elapsedNs ? static_cast<uint64_t>(static_cast<long double>(recvDelta) * 1000000000.0L / elapsedNs) : 0)
        << "}\n";
  ++g_sampleCounts[comm];
  g_log.flush();
}

void rcclCcldSampleDiagnostics(const struct ncclComm* comm, uint64_t timestampNs,
                               uint64_t sampleIntervalNs) {
  if (!comm || !enabled() || !comm->ccldDiag) return;
  struct Snapshot { uint64_t counter, send, recv, sendDelta, recvDelta, elapsed; uint16_t channel; bool reset; };
  std::vector<Snapshot> snapshots;
  {
    std::lock_guard<std::mutex> guard(g_lock);
    auto pit = g_pools.find(comm);
    auto oit = g_operations.find(comm);
    if (pit == g_pools.end() || oit == g_operations.end()) return;
    for (const auto& entry : oit->second) {
      uint32_t index = UINT32_MAX;
      if (!pit->second->find(entry.first, &index)) continue;
      volatile struct ncclCcldDiag* diag = comm->ccldDiag + index;
      if (__atomic_load_n(&diag->header.mode, __ATOMIC_ACQUIRE) == 0) continue;
      for (uint16_t channel : entry.second.channels) {
        std::string key = std::to_string(reinterpret_cast<uintptr_t>(comm)) + ":" +
                          std::to_string(entry.first) + ":" + std::to_string(channel);
        volatile rccl_ccld::ChannelBody* body = &diag->channels[channel];
        const uint64_t sends = __atomic_load_n(&body->sendCount, __ATOMIC_RELAXED);
        const uint64_t recvs = __atomic_load_n(&body->recvCount, __ATOMIC_RELAXED);
        auto prev = g_samplePrevious.find(key);
        const bool reset = prev != g_samplePrevious.end() &&
                           (sends < prev->second.send || recvs < prev->second.recv);
        const uint64_t elapsed = prev == g_samplePrevious.end() || reset ? 0 : timestampNs - prev->second.timestamp;
        if (prev == g_samplePrevious.end() || elapsed >= sampleIntervalNs || reset) {
          const uint64_t sendDelta = prev == g_samplePrevious.end() || reset ? 0 : sends - prev->second.send;
          const uint64_t recvDelta = prev == g_samplePrevious.end() || reset ? 0 : recvs - prev->second.recv;
          snapshots.push_back({entry.first, sends, recvs, sendDelta, recvDelta, elapsed, channel, reset});
          g_samplePrevious[key] = {timestampNs, sends, recvs};
        }
      }
    }
  }
  for (const Snapshot& item : snapshots)
    rcclCcldRecordDeviceSample(comm, item.counter, item.channel, item.send, item.recv,
                               timestampNs, item.sendDelta, item.recvDelta, item.elapsed, item.reset);
}

