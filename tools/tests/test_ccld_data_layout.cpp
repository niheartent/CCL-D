#include "ccld_data.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

int main() {
  using rccl_ccld::ChannelBody;
  using rccl_ccld::Header;
  using rccl_ccld::Record;
  using rccl_ccld::TraceId;

  static_assert(sizeof(TraceId) == 60);
  static_assert(sizeof(Header) == 96);
  static_assert(offsetof(Record<4>, channels) == sizeof(Header));
  static_assert(sizeof(Record<4>) == sizeof(Header) + 4 * sizeof(ChannelBody));

  TraceId id = rccl_ccld::makeTraceId(0x0123456789abcdefULL, 17);
  for (uint8_t byte : id.extension) assert(byte == 0);

  Record<3> record{};
  record.header.kernelIndex = 5;
  record.header.mode = 1;
  record.header.counter = id.counter;
  record.header.detectChannel = 4;
  record.header.traceId = id;
  record.channels[2].sendCount = 19;
  record.channels[2].recvCount = 23;
  assert(record.header.traceId.commId == id.commId);
  assert(record.header.traceId.counter == record.header.counter);
  assert(record.channels[0].sendCount == 0);
  assert(record.channels[2].sendCount == 19);
  assert(record.channels[2].recvCount == 23);
  std::cout << "CCL-D data layout PASS\n";
}
