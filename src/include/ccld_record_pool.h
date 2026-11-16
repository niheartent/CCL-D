#ifndef RCCL_CCLD_RECORD_POOL_H_
#define RCCL_CCLD_RECORD_POOL_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace rccl_ccld {

// Host-side LRU selects only free frame blocks. Under pressure the observation
// is dropped; communication never waits and active data is never overwritten.
class RecordPool {
 public:
  explicit RecordPool(size_t capacity) : slots_(capacity) {}

  uint64_t nextCounter() {
    std::lock_guard<std::mutex> lock(mutex_);
    return nextCounter_++;
  }

  bool reserve(uint64_t counter, uint32_t* blockIndex) {
    if (!blockIndex || slots_.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t selected = slots_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].counter == counter) {
        selected = i;
        break;
      }
      if (!slots_[i].active && slots_[i].lastUse < oldest) {
        selected = i;
        oldest = slots_[i].lastUse;
      }
    }
    if (selected == slots_.size()) return false;
    Slot& slot = slots_[selected];
    slot.counter = counter;
    slot.active = true;
    slot.lastUse = ++clock_;
    *blockIndex = static_cast<uint32_t>(selected);
    return true;
  }

  bool release(uint64_t counter) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Slot& slot : slots_) {
      if (slot.active && slot.counter == counter) {
        slot.active = false;
        slot.counter = UINT64_MAX;
        slot.lastUse = ++clock_;
        return true;
      }
    }
    return false;
  }

  bool find(uint64_t counter, uint32_t* blockIndex) const {
    if (!blockIndex) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].counter == counter) {
        *blockIndex = static_cast<uint32_t>(i);
        return true;
      }
    }
    return false;
  }

  size_t capacity() const { return slots_.size(); }

 private:
  struct Slot { uint64_t counter = UINT64_MAX; uint64_t lastUse = 0; bool active = false; };
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  uint64_t clock_ = 0;
  uint64_t nextCounter_ = 0;
};

}  // namespace rccl_ccld

#endif
