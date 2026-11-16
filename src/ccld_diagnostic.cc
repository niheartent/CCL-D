#include "ccld_diagnostic.h"
#include "collectives.h"
#include "comm.h"
#include "nccl.h"
#include "device.h"
#include "ccld_record_pool.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr int kDefaultCcldDiagSlots = 16;
std::mutex g_lock;
std::ofstream g_log;
std::string g_path;
std::atomic<bool> g_initialized{false};
std::unordered_map<const ncclComm*, std::unique_ptr<rccl_ccld::RecordPool>> g_pools;
struct ParentRecord {
  const ncclComm* comm = nullptr;
  uint64_t counter = UINT64_MAX;
  uint32_t kernelIndex = UINT32_MAX;
  uint32_t refs = 0;
};
thread_local std::vector<ParentRecord> g_parentRecordStack;

bool enabled() {
  const char* value = std::getenv("RCCL_CCLD_ENABLE");
  if (!(value && (std::string(value) == "1" || std::string(value) == "ON" || std::string(value) == "on"))) return false;
  const char* mode = std::getenv("RCCL_CCLD_MODE");
  if (mode && *mode) {
    char* end = nullptr;
    const long parsed = std::strtol(mode, &end, 10);
    if (end != mode && *end == '\0' && parsed == 0) return false;
  }
  return true;
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
  const bool hasTemplate = path.find("{rank}") != std::string::npos ||
                           path.find("${RANK}") != std::string::npos ||
                           path.find("%r") != std::string::npos;
  replaceAll(path, "{rank}", rankText);
  replaceAll(path, "${RANK}", rankText);
  replaceAll(path, "%r", rankText);
  if (!hasTemplate) {
    const size_t slash = path.find_last_of('/');
    const size_t dot = path.find_last_of('.');
    const std::string suffix = ".rank" + rankText;
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
      path.insert(dot, suffix);
    } else {
      path += suffix;
    }
  }
  return path;
}

void init(const ncclComm* comm) {
  if (g_initialized.exchange(true)) return;
  const char* configured = std::getenv("RCCL_CCLD_LOG");
  g_path = rankPath(configured, comm ? comm->rank : -1);
  g_log.open(g_path, std::ios::out | std::ios::app);
}

uint64_t nowNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

const char* safe(const char* value) { return value ? value : ""; }

int envInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) return fallback;
  return static_cast<int>(parsed);
}

bool faultSelected(const ncclComm* comm, uint64_t counter) {
  const int rank = envInt("RCCL_CCLD_FAULT_RANK", 0);
  const int selectedCounter = envInt("RCCL_CCLD_FAULT_COUNTER", -1);
  return comm->rank == rank && (selectedCounter < 0 || counter == static_cast<uint64_t>(selectedCounter));
}
}

bool rcclCcldEnabled() { return enabled(); }

void rcclCcldRecordEnqueue(struct ncclInfo* info) {
  if (!info || !info->comm || !enabled()) return;
  info->ccldCounter = UINT64_MAX;
  if (info->comm->nRanks <= 1 || info->count == 0) return;
  if (!g_parentRecordStack.empty() && g_parentRecordStack.back().comm == info->comm &&
      g_parentRecordStack.back().counter != UINT64_MAX) {
    info->ccldCounter = g_parentRecordStack.back().counter;
    info->ccldKernelIndex = g_parentRecordStack.back().kernelIndex;
    ++g_parentRecordStack.back().refs;
    return;
  }
  std::lock_guard<std::mutex> guard(g_lock);
  init(info->comm);
  if (!g_log) return;
  const ncclComm* comm = info->comm;
  size_t count = info->count;
  size_t bytes = count * ncclTypeSize(info->datatype);
  int nranks = comm->nRanks;
  const char* op = safe(info->opName);
  auto poolIt = g_pools.find(comm);
  if (poolIt == g_pools.end()) {
    const size_t slots = comm->ccldDiagSlots > 0 ? static_cast<size_t>(comm->ccldDiagSlots) : kDefaultCcldDiagSlots;
    poolIt = g_pools.emplace(comm, std::unique_ptr<rccl_ccld::RecordPool>(new rccl_ccld::RecordPool(slots))).first;
  }
  const uint64_t counter = poolIt->second->nextCounter();
  info->ccldCounter = counter;
  const char* fault = std::getenv("RCCL_CCLD_FAULT");
  const bool inject = fault && faultSelected(comm, counter);
  if (inject && std::string(fault) == "skip_record") {
    info->ccldKernelIndex = UINT32_MAX;
    return;
  }
  uint32_t kernelIndex = UINT32_MAX;
  const bool reserved = poolIt->second->reserve(counter, 0, &kernelIndex) == rccl_ccld::ReserveResult::Reserved;
  if (!reserved) kernelIndex = UINT32_MAX;
  if (reserved) poolIt->second->markRunning(kernelIndex);
  info->ccldKernelIndex = kernelIndex;
  if (inject && std::string(fault) == "op_mismatch") op = "CCLD_INJECTED_OP_MISMATCH";
  if (inject && std::string(fault) == "size_mismatch") {
    ++count;
    bytes += ncclTypeSize(info->datatype);
  }
  // This test-only mode creates a diagnostic-only missing-rank condition.
  // The communicator and the collective call itself remain untouched.
  if (inject && std::string(fault) == "missing_rank") ++nranks;
  std::ostringstream trace;
  trace << std::hex << std::setw(16) << std::setfill('0') << comm->commHash
        << ':' << std::dec << counter;
  g_log << "{\"schema\":1,\"phase\":\"enqueue\",\"timestamp_ns\":" << nowNs()
        << ",\"trace_id\":\"" << trace.str()
        << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"counter\":" << counter
        << ",\"op_count\":" << comm->opCount
        << ",\"rank\":" << comm->rank << ",\"nranks\":" << nranks
        << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
        << ",\"op\":\"" << op << "\",\"count\":" << count
        << ",\"bytes\":" << bytes << ",\"datatype\":" << static_cast<int>(info->datatype)
        << ",\"root\":" << info->root << ",\"kernel_index\":";
  if (kernelIndex == UINT32_MAX) g_log << "null";
  else g_log << kernelIndex;
  g_log << ",\"diagnostic_status\":\"" << (reserved ? "RECORDED" : "RESOURCE_EXHAUSTED") << "\"}\n";
  g_log.flush();
}

bool rcclCcldBeginParent(struct ncclInfo* info) {
  if (!info || !info->comm || !enabled() || !g_parentRecordStack.empty()) return false;
  rcclCcldRecordEnqueue(info);
  // Keep the parent identity even when no device slot was available. Otherwise
  // each expanded AllToAll Send/Recv child would allocate a fresh counter and
  // shift this rank's TraceID sequence relative to its peers.
  if (info->ccldCounter == UINT64_MAX) return false;
  g_parentRecordStack.push_back(ParentRecord{info->comm, info->ccldCounter, info->ccldKernelIndex, 0});
  return true;
}

void rcclCcldEndParent() {
  if (!g_parentRecordStack.empty()) g_parentRecordStack.pop_back();
}

void rcclCcldRegisterBody(const struct ncclComm* comm, uint64_t counter,
                          uint64_t relatedCounter, uint16_t channel) {
  if (!comm || !enabled() || counter == UINT64_MAX) return;
  std::lock_guard<std::mutex> guard(g_lock);
  auto it = g_pools.find(comm);
  if (it == g_pools.end()) return;
  uint32_t index = UINT32_MAX;
  if (it->second->find(counter, &index)) {
    it->second->registerBody(counter, index, channel);
  }
  if (relatedCounter != UINT64_MAX && relatedCounter != counter) {
    uint32_t relatedIndex = UINT32_MAX;
    if (it->second->find(relatedCounter, &relatedIndex))
      it->second->registerBody(relatedCounter, relatedIndex, channel);
  }
}

void rcclCcldFinalizeDiagnostics(const struct ncclComm* comm) {
  if (!comm || !enabled() || !comm->ccldDiag) return;
  const int slots = comm->ccldDiagSlots > 0 ? comm->ccldDiagSlots : kDefaultCcldDiagSlots;
  for (int channel = 0; channel < MAXCHANNELS; ++channel) {
    for (int slot = 0; slot < slots; ++slot) {
      struct ncclCcldDiag* diag = comm->ccldDiag + channel * slots + slot;
      if (__atomic_load_n(&diag->state, __ATOMIC_ACQUIRE) == 1)
        __atomic_store_n(&diag->state, 2u, __ATOMIC_RELEASE);
    }
  }
}

void rcclCcldForgetComm(const struct ncclComm* comm) {
  if (!comm) return;
  std::lock_guard<std::mutex> guard(g_lock);
  g_pools.erase(comm);
}

void rcclCcldRecordDeviceDiag(const struct ncclComm* comm,
                              const struct ncclCcldDiag* diag) {
  if (!comm || !diag || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(comm);
  if (!g_log) {
    auto poolIt = g_pools.find(comm);
    if (poolIt != g_pools.end()) {
      uint32_t index = UINT32_MAX;
      if (poolIt->second->find(diag->ccldCounter, &index))
        poolIt->second->consumeBody(diag->ccldCounter, index);
      if (diag->relatedCcldCounter != UINT64_MAX) {
        uint32_t relatedIndex = UINT32_MAX;
        if (poolIt->second->find(diag->relatedCcldCounter, &relatedIndex))
          poolIt->second->consumeBody(diag->relatedCcldCounter, relatedIndex);
      }
    }
    return;
  }
  std::ostringstream traceId;
  traceId << std::hex << std::setw(16) << std::setfill('0') << comm->commHash
          << ':' << std::dec << diag->ccldCounter;
  // Diagnostic-only fault modes make hang/slow handling reproducible inside
  // the container without touching GPU data movement.  They deliberately
  // alter only the emitted evidence for a selected rank/counter.
  const char* fault = std::getenv("RCCL_CCLD_FAULT");
  const bool inject = fault && faultSelected(comm, diag->ccldCounter);
  const std::string faultName = fault ? fault : "";
  bool complete = true;
  uint64_t startTimestamp = diag->startTimestamp;
  uint64_t endTimestamp = diag->endTimestamp;
  if (inject && (faultName == "completion_missing" || faultName == "missing_completion" ||
                 faultName == "stall" || faultName == "hang")) complete = false;
  if (inject && faultName == "slow" && endTimestamp > startTimestamp) {
    const uint64_t duration = endTimestamp - startTimestamp;
    endTimestamp = startTimestamp + duration * 10;
  }
  g_log << "{\"schema\":1,\"phase\":\"complete\",\"timestamp_ns\":" << nowNs()
        << ",\"trace_id\":\"" << traceId.str()
        << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"counter\":" << diag->ccldCounter
        << ",\"related_counter\":";
  if (diag->relatedCcldCounter == UINT64_MAX) g_log << "null";
  else g_log << diag->relatedCcldCounter;
  g_log
        << ",\"dropped_batches\":" << diag->droppedBatches
        << ",\"op_count\":" << diag->opCount
        << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
        << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
        << ",\"channel\":" << diag->channel
        << ",\"algorithm\":" << static_cast<int>(diag->algorithm)
        << ",\"protocol\":" << static_cast<int>(diag->protocol)
        << ",\"send_count\":" << diag->sendCount
        << ",\"recv_count\":" << diag->recvCount
        << ",\"send_count_by_protocol\":[" << diag->sendCountByProto[0] << ","
        << diag->sendCountByProto[1] << "," << diag->sendCountByProto[2] << "]"
        << ",\"recv_count_by_protocol\":[" << diag->recvCountByProto[0] << ","
        << diag->recvCountByProto[1] << "," << diag->recvCountByProto[2] << "]"
        << ",\"send_peers\":[";
  for (int i = 0; i < diag->sendPeerCount; ++i) {
    if (i) g_log << ',';
    g_log << "{\"peer\":" << diag->sendPeerIds[i]
          << ",\"bytes\":" << diag->sendBytesByPeer[i] << "}";
  }
  g_log << "],\"recv_peers\":[";
  for (int i = 0; i < diag->recvPeerCount; ++i) {
    if (i) g_log << ',';
    g_log << "{\"peer\":" << diag->recvPeerIds[i]
          << ",\"bytes\":" << diag->recvBytesByPeer[i] << "}";
  }
  g_log << "],\"peer_overflow\":" << (diag->peerOverflow ? "true" : "false")
        << ",\"send_peer_protocol_bytes\":[";
  for (int proto = 0; proto < 3; ++proto) {
    if (proto) g_log << ',';
    g_log << '[';
    for (int i = 0; i < diag->sendPeerCount; ++i) {
      if (i) g_log << ',';
      g_log << diag->sendBytesByProtocolPeer[proto][i];
    }
    g_log << ']';
  }
  g_log << "],\"recv_peer_protocol_bytes\":[";
  for (int proto = 0; proto < 3; ++proto) {
    if (proto) g_log << ',';
    g_log << '[';
    for (int i = 0; i < diag->recvPeerCount; ++i) {
      if (i) g_log << ',';
      g_log << diag->recvBytesByProtocolPeer[proto][i];
    }
    g_log << ']';
  }
  g_log << ']'
        << ",\"start_timestamp\":" << startTimestamp
        << ",\"end_timestamp\":" << endTimestamp
        << ",\"complete\":" << (complete ? "true" : "false");
  if (diag->algorithm == NCCL_ALGO_TREE && diag->channel < MAXCHANNELS) {
    const ncclTree& tree = comm->channels[diag->channel].tree;
    g_log << ",\"tree_up\":" << tree.up
          << ",\"tree_down0\":" << tree.down[0]
          << ",\"tree_down1\":" << tree.down[1]
          << ",\"tree_down2\":" << tree.down[2]
          << ",\"tree_depth\":" << tree.depth;
  } else if (diag->algorithm == NCCL_ALGO_RING && diag->channel < MAXCHANNELS) {
    const ncclRing& ring = comm->channels[diag->channel].ring;
    g_log << ",\"ring_prev\":" << ring.prev
          << ",\"ring_next\":" << ring.next;
  }
  g_log << "}\n";
  g_log.flush();
  auto poolIt = g_pools.find(comm);
  if (poolIt != g_pools.end()) {
    uint32_t index = UINT32_MAX;
    if (poolIt->second->find(diag->ccldCounter, &index))
      poolIt->second->consumeBody(diag->ccldCounter, index);
    if (diag->relatedCcldCounter != UINT64_MAX) {
      uint32_t relatedIndex = UINT32_MAX;
      if (poolIt->second->find(diag->relatedCcldCounter, &relatedIndex))
        poolIt->second->consumeBody(diag->relatedCcldCounter, relatedIndex);
    }
  }
}

void rcclCcldRecordDeviceSample(const struct ncclComm* comm, uint64_t opCount,
                               uint16_t channel, uint64_t sendCount,
                               uint64_t recvCount, uint64_t timestampNs,
                               uint64_t sendDelta, uint64_t recvDelta,
                               uint64_t elapsedNs, bool counterReset) {
  if (!comm || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(comm);
  if (!g_log) return;
  std::ostringstream traceId;
  traceId << std::hex << std::setw(16) << std::setfill('0') << comm->commHash
          << ':' << std::dec << opCount;
  g_log << "{\"schema\":1,\"phase\":\"sample\",\"timestamp_ns\":" << timestampNs
        << ",\"trace_id\":\"" << traceId.str()
        << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"counter\":" << opCount << ",\"op_count\":" << opCount
        << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
        << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
        << ",\"channel\":" << channel << ",\"send_count\":" << sendCount
        << ",\"recv_count\":" << recvCount
        << ",\"send_delta\":" << sendDelta << ",\"recv_delta\":" << recvDelta
        << ",\"elapsed_ns\":" << elapsedNs
        << ",\"counter_reset\":" << (counterReset ? "true" : "false")
        << ",\"send_rate_Bps\":" << (elapsedNs ? static_cast<uint64_t>(
              static_cast<long double>(sendDelta) * 1000000000.0L / elapsedNs) : 0)
        << ",\"recv_rate_Bps\":" << (elapsedNs ? static_cast<uint64_t>(
              static_cast<long double>(recvDelta) * 1000000000.0L / elapsedNs) : 0)
        << ",\"state\":\"running\"}\n";
  g_log.flush();
}

#ifdef ENABLE_COLLTRACE
void rcclCcldRecordDevice(const struct ncclComm* comm,
                          const struct ncclCollTrace* trace,
                          uint64_t send_count, uint64_t recv_count,
                          bool complete) {
  if (!comm || !trace || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(comm);
  if (!g_log) return;
  const uint64_t counter = trace->opCount;
  std::ostringstream traceId;
  traceId << std::hex << std::setw(16) << std::setfill('0') << comm->commHash
          << ':' << std::dec << counter;
  g_log << "{\"schema\":1,\"phase\":\"complete\",\"timestamp_ns\":" << nowNs()
        << ",\"trace_id\":\"" << traceId.str()
        << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"counter\":" << counter
        << ",\"op_count\":" << counter
        << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
        << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
        << ",\"channel\":" << static_cast<int>(trace->channelId)
        << ",\"send_count\":" << send_count
        << ",\"recv_count\":" << recv_count
        << ",\"topology_kind\":\""
        << ((trace->type & ncclCollTraceP2pElemType) ? "p2p" : "collective")
        << "\""
        << ((trace->type & ncclCollTraceP2pElemType)
              ? (std::string(",\"send_rank\":") + std::to_string(trace->p2p.sendRank) +
                 ",\"recv_rank\":" + std::to_string(trace->p2p.recvRank) +
                 ",\"channel_base\":" + std::to_string(trace->p2p.channelBase) +
                 ",\"send_channels\":" + std::to_string(trace->p2p.nSendChannels) +
                 ",\"recv_channels\":" + std::to_string(trace->p2p.nRecvChannels))
              : std::string())
        << ",\"complete\":" << (complete ? "true" : "false") << "}\n";
  g_log.flush();
}
#endif
