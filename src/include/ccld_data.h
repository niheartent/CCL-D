#ifndef RCCL_CCLD_DATA_H_
#define RCCL_CCLD_DATA_H_

#include <cstddef>
#include <cstdint>

namespace rccl_ccld {

// The reference diagram defines the TraceID wire layout as 8-byte commId,
// 8-byte counter and 44-byte extension. The extension has no documented
// subfields, so callers must initialize it to zero and must not infer meaning.
struct __attribute__((packed)) TraceId {
  uint64_t commId;
  uint64_t counter;
  uint8_t extension[44];
};
static_assert(sizeof(TraceId) == 60, "reference TraceID is 60 bytes");
static_assert(offsetof(TraceId, commId) == 0, "TraceID commId offset");
static_assert(offsetof(TraceId, counter) == 8, "TraceID counter offset");
static_assert(offsetof(TraceId, extension) == 16, "TraceID extension offset");

inline TraceId makeTraceId(uint64_t commId, uint64_t counter) {
  return TraceId{commId, counter, {0}};
}

// The diagnostic-space diagram uses 8-byte KID/mode/counter/detectChannel
// words followed by TraceID. TraceID is 60 bytes: commId, counter and the
// explicitly reserved 44-byte extension.
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
struct alignas(8) Record {
  Header header;
  ChannelBody channels[ChannelCount];
};

}  // namespace rccl_ccld

#endif
