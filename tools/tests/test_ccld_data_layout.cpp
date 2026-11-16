#include "ccld_data.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

int main() {
  using rccl_ccld::ChannelBody;
  using rccl_ccld::Header;
  using rccl_ccld::Block;
  using rccl_ccld::Frame;
  using rccl_ccld::TraceId;

  static_assert(sizeof(TraceId) == 60);
  static_assert(sizeof(Header) == 96);
  static_assert(offsetof(Block<4>, channels) == sizeof(Header));
  static_assert(sizeof(Block<4>) == sizeof(Header) + 4 * sizeof(ChannelBody));

  TraceId id = rccl_ccld::makeTraceId(0x0123456789abcdefULL, 17);
  for (uint8_t byte : id.reserved) assert(byte == 0);

  Block<3> record{};
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
  Frame<2, 3> frame{};
  frame.blocks[1].header.counter = 9;
  frame.blocks[1].channels[2].recvCount = 31;
  assert((rccl_ccld::blockIndex<2, 3>(9) == 1));
  assert(rccl_ccld::channelBodyOffset<3>(2) == sizeof(Header) + 2 * sizeof(ChannelBody));
  assert(frame.blocks[1].channels[2].recvCount == 31);
  std::cout << "CCL-D data layout PASS\n";
}
