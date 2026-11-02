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
#include <unordered_map>

namespace {
std::mutex g_lock;
std::ofstream g_log;
std::string g_path;
std::atomic<bool> g_initialized{false};
std::unordered_map<const ncclComm*, uint64_t> g_counters;

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
  // diagnostic counter at enqueue time so every submitted collective gets a
  // distinct trace even when several operations are queued together.
  const uint64_t counter = g_counters[comm]++;
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
  g_counters.erase(comm);
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
  g_log << "{\"schema\":1,\"phase\":\"complete\",\"timestamp_ns\":" << nowNs()
        << ",\"trace_id\":\"" << traceId.str()
        << "\",\"comm_id\":\"" << std::hex << comm->commHash << std::dec
        << "\",\"counter\":" << diag->opCount
        << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
        << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
        << ",\"channel\":" << diag->channel
        << ",\"algorithm\":" << static_cast<int>(diag->algorithm)
        << ",\"protocol\":" << static_cast<int>(diag->protocol)
        << ",\"send_count\":" << diag->sendCount
        << ",\"recv_count\":" << diag->recvCount
        << ",\"start_timestamp\":" << diag->startTimestamp
        << ",\"end_timestamp\":" << diag->endTimestamp
        << ",\"complete\":true}\n";
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
        << ",\"rank\":" << comm->rank << ",\"nranks\":" << comm->nRanks
        << ",\"node\":" << comm->node << ",\"nnodes\":" << comm->nNodes
        << ",\"channel\":" << static_cast<int>(trace->channelId)
        << ",\"send_count\":" << send_count
        << ",\"recv_count\":" << recv_count
        << ",\"complete\":" << (complete ? "true" : "false") << "}\n";
  g_log.flush();
}
#endif
