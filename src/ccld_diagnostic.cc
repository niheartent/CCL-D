#include "ccld_diagnostic.h"
#include "collectives.h"
#include "comm.h"
#include "nccl.h"
#include "device.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

namespace {
std::mutex g_lock;
std::ofstream g_log;
std::string g_path;
std::atomic<bool> g_initialized{false};

bool enabled() {
  const char* value = std::getenv("RCCL_CCLD_ENABLE");
  return value && (std::string(value) == "1" || std::string(value) == "ON" || std::string(value) == "on");
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

void rcclCcldRecordEnqueue(const struct ncclInfo* info) {
  if (!info || !info->comm || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(info->comm);
  if (!g_log) return;
  const ncclComm* comm = info->comm;
  size_t count = info->count;
  size_t bytes = count * ncclTypeSize(info->datatype);
  int nranks = comm->nRanks;
  const char* op = safe(info->opName);
  // RCCL's internal opCount advances later in the proxy path.  Keep a
  // Use RCCL's canonical operation counter so enqueue and device completion
  // resolve to the same trace slot. RCCL advances it in the scheduling path.
  const uint64_t counter = comm->opCount;
  const char* fault = std::getenv("RCCL_CCLD_FAULT");
  const bool inject = fault && faultSelected(comm, counter);
  if (inject && std::string(fault) == "skip_record") return;
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
        << ",\"root\":" << info->root << "}\n";
  g_log.flush();
}

void rcclCcldForgetComm(const struct ncclComm* comm) {
  if (!comm) return;
  std::lock_guard<std::mutex> guard(g_lock);
  // No diagnostic-side counter is retained; this avoids host/device drift.
}

void rcclCcldRecordDeviceDiag(const struct ncclComm* comm,
                              const struct ncclCcldDiag* diag) {
  if (!comm || !diag || !enabled()) return;
  std::lock_guard<std::mutex> guard(g_lock);
  init(comm);
  if (!g_log) return;
  std::ostringstream traceId;
  traceId << std::hex << std::setw(16) << std::setfill('0') << comm->commHash
          << ':' << std::dec << diag->opCount;
  // Diagnostic-only fault modes make hang/slow handling reproducible inside
  // the container without touching GPU data movement.  They deliberately
  // alter only the emitted evidence for a selected rank/counter.
  const char* fault = std::getenv("RCCL_CCLD_FAULT");
  const bool inject = fault && faultSelected(comm, diag->opCount);
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
        << "\",\"counter\":" << diag->opCount
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
