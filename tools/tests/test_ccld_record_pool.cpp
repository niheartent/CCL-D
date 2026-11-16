#include "ccld_record_pool.h"
#include <cassert>
#include <cstdint>
#include <iostream>

using rccl_ccld::RecordPool;
using rccl_ccld::RecordState;
using rccl_ccld::ReserveResult;

int main() {
  const uint64_t flushCounters[] = {41, 43, 47};
  assert(rccl_ccld::containsCounter(flushCounters, 3, 43));
  assert(!rccl_ccld::containsCounter(flushCounters, 3, 42));
  assert(!rccl_ccld::containsCounter(nullptr, 0, 41));

  RecordPool pool(2);
  uint32_t a = UINT32_MAX, b = UINT32_MAX, c = UINT32_MAX;

  // Multiple host API calls in one RCCL group still receive distinct TraceIDs.
  assert(pool.nextCounter() == 0);
  assert(pool.nextCounter() == 1);

  assert(pool.reserve(10, 2, &a) == ReserveResult::Reserved);
  assert(pool.reserve(11, 1, &b) == ReserveResult::Reserved);
  assert(a != b);
  assert(pool.reserve(12, 1, &c) == ReserveResult::ResourceExhausted);

  // Active records are never overwritten when capacity is exhausted.
  rccl_ccld::RecordView view;
  assert(pool.view(a, &view) && view.counter == 10 && view.state == RecordState::Reserved);
  assert(pool.markRunning(a));
  assert(pool.registerBody(10, a, 0));
  assert(pool.consumeBody(10, a));

  // A released body is reusable; stale lifecycle transitions are rejected.
  assert(pool.reserve(12, 1, &c) == ReserveResult::Reserved);
  assert(c == a);
  assert(pool.markRunning(c));
  assert(pool.registerBody(12, c, 0));
  assert(pool.consumeBody(12, c));

  // A multi-channel TraceID stays pinned until every registered body is
  // consumed by the diagnostic thread.
  assert(pool.reserve(13, 0, &c) == ReserveResult::Reserved);
  assert(pool.markRunning(c));
  assert(pool.registerBody(13, c, 1));
  assert(pool.registerBody(13, c, 1)); // duplicate channel is ignored
  assert(pool.registerBody(13, c, 2));
  assert(!pool.consumeBody(12, c));
  // Each registered channel body is consumed once after its callback.
  assert(pool.consumeBody(13, c));
  assert(pool.view(c, &view) && view.state == RecordState::Running);
  assert(pool.reserve(14, 0, &b) == ReserveResult::ResourceExhausted);
  assert(pool.consumeBody(13, c));
  assert(pool.reserve(14, 0, &b) == ReserveResult::Reserved);
  assert(!pool.consumeBody(13, c));

  // Paired P2P send/recv operations have separate counters but share one
  // device completion; both records become reusable after that completion.
  RecordPool p2pPool(2);
  uint32_t sendIndex = UINT32_MAX, recvIndex = UINT32_MAX;
  assert(p2pPool.reserve(20, 0, &sendIndex) == ReserveResult::Reserved);
  assert(p2pPool.reserve(21, 0, &recvIndex) == ReserveResult::Reserved);
  assert(p2pPool.markRunning(sendIndex));
  assert(p2pPool.markRunning(recvIndex));
  assert(p2pPool.registerBody(20, sendIndex, 0));
  assert(p2pPool.registerBody(21, recvIndex, 0));
  assert(p2pPool.consumeBody(20, sendIndex));
  assert(p2pPool.consumeBody(21, recvIndex));
  assert(p2pPool.reserve(22, 0, &sendIndex) == ReserveResult::Reserved);

  assert(!pool.markRunning(999));
  std::cout << "record pool lifecycle PASS\n";
  return 0;
}
