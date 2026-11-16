#ifndef RCCL_CCLD_DATA_H_
#define RCCL_CCLD_DATA_H_

#include <cstddef>
#include <cstdint>

namespace rccl_ccld {

// Storage layout from the reference diagram. The 44-byte reserved area has no
// assigned fields or runtime semantics at present.
struct __attribute__((packed)) TraceId {
  uint64_t commId;
  uint64_t counter;
  uint8_t reserved[44];
};
static_assert(sizeof(TraceId) == 60, "reference TraceID is 60 bytes");
static_assert(offsetof(TraceId, commId) == 0, "TraceID commId offset");
static_assert(offsetof(TraceId, counter) == 8, "TraceID counter offset");
static_assert(offsetof(TraceId, reserved) == 16, "TraceID reserved offset");

inline TraceId makeTraceId(uint64_t commId, uint64_t counter) {
  return TraceId{commId, counter, {0}};
}

// Header and body for one KID block. KID is selected by counter modulo the
// configured block count; each channel contributes exactly two 64-bit counts.
struct alignas(8) Header {
  uint64_t kernelIndex;
  uint64_t mode;
  uint64_t counter;
  uint64_t detectChannel;
  TraceId traceId;
};
static_assert(offsetof(Header, kernelIndex) == 0, "header KID offset");
static_assert(offsetof(Header, mode) == 8, "header mode offset");
static_assert(offsetof(Header, counter) == 16, "header counter offset");
static_assert(offsetof(Header, detectChannel) == 24, "header channel offset");
static_assert(offsetof(Header, traceId) == 32, "header TraceID offset");
static_assert(sizeof(Header) == 96, "header plus 60-byte TraceID, 8-byte aligned");

// The reference body contains only the two per-channel counters.
struct alignas(8) ChannelBody {
  uint64_t sendCount;
  uint64_t recvCount;
};
static_assert(sizeof(ChannelBody) == 16, "channel body is two 64-bit counters");

template <size_t ChannelCount>
struct alignas(8) Block {
  Header header;
  ChannelBody channels[ChannelCount];
};

template <size_t BlockCount, size_t ChannelCount>
struct alignas(8) Frame {
  static_assert(BlockCount > 0, "a diagnostic frame needs at least one block");
  static_assert(ChannelCount > 0, "a diagnostic frame needs at least one channel");
  Block<ChannelCount> blocks[BlockCount];
};

template <size_t ChannelCount>
inline size_t channelBodyOffset(size_t channel) {
  return sizeof(Header) + channel * sizeof(ChannelBody);
}

template <size_t BlockCount, size_t ChannelCount>
inline size_t blockIndex(uint64_t counter) {
  static_assert(BlockCount > 0, "a diagnostic frame needs at least one block");
  return static_cast<size_t>(counter % BlockCount);
}

}  // namespace rccl_ccld

#endif
