#include "ccld_record_pool.h"
#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
  rccl_ccld::RecordPool pool(2);
  uint32_t first = UINT32_MAX, second = UINT32_MAX, third = UINT32_MAX;
  const uint64_t a = pool.nextCounter();
  const uint64_t b = pool.nextCounter();
  const uint64_t c = pool.nextCounter();
  assert(a == 0 && b == 1 && c == 2);
  assert(pool.reserve(a, &first));
  assert(pool.reserve(b, &second));
  assert(first != second);
  assert(pool.find(a, &third) && third == first);
  assert(!pool.reserve(c, &third)); // active blocks are never overwritten
  assert(pool.release(a));
  assert(pool.reserve(c, &third));
  assert(third == first);
  assert(!pool.find(a, &first));
  assert(pool.find(b, &first) && first == second);
  assert(pool.find(c, &first) && first == third);
  assert(!pool.reserve(c, nullptr));
  rccl_ccld::RecordPool empty(0);
  assert(!empty.reserve(1, &first));
  std::cout << "frame block LRU PASS\n";
  return 0;
}
