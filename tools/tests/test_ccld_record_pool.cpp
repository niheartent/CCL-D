#include "ccld_record_pool.h"
#include <cassert>
#include <cstdint>
#include <iostream>

using rccl_ccld::RecordPool;
using rccl_ccld::RecordState;
using rccl_ccld::ReserveResult;

int main() {
  RecordPool pool(2);
  uint32_t a = UINT32_MAX, b = UINT32_MAX, c = UINT32_MAX;

  assert(pool.reserve(10, 2, &a) == ReserveResult::Reserved);
  assert(pool.reserve(11, 1, &b) == ReserveResult::Reserved);
  assert(a != b);
  assert(pool.reserve(12, 1, &c) == ReserveResult::ResourceExhausted);

  // Active records are never overwritten when capacity is exhausted.
  rccl_ccld::RecordView view;
  assert(pool.view(a, &view) && view.counter == 10 && view.state == RecordState::Reserved);
  assert(pool.markRunning(a));
  assert(pool.markChannelComplete(a, 0));
  assert(!pool.markPlanComplete(a));
  assert(pool.markChannelComplete(a, 1));
  assert(!pool.markChannelComplete(a, 1));
  assert(pool.markPlanComplete(a));
  assert(pool.markConsumed(a));
  assert(pool.release(a));

  // A released body is reusable; stale lifecycle transitions are rejected.
  assert(pool.reserve(12, 1, &c) == ReserveResult::Reserved);
  assert(c == a);
  assert(!pool.markConsumed(c));
  assert(pool.markRunning(c));
  assert(pool.markMissingCompletion(c));
  assert(pool.release(c));

  assert(!pool.markRunning(999));
  assert(pool.reserve(13, 0, &c) == ReserveResult::ResourceExhausted);
  std::cout << "record pool lifecycle PASS\n";
  return 0;
}
