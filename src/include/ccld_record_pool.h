#ifndef RCCL_CCLD_RECORD_POOL_H_
#define RCCL_CCLD_RECORD_POOL_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace rccl_ccld {

inline bool containsCounter(const uint64_t* counters, size_t count, uint64_t counter) {
  for (size_t i = 0; i < count; ++i) if (counters[i] == counter) return true;
  return false;
}

// Minimal host lifetime for pinned diagnostic bodies; planner state is not
// modeled here. A body is registered once per channel and consumed once.
enum class RecordState : uint8_t { Free, Reserved, Running };
enum class ReserveResult : uint8_t { Reserved, ResourceExhausted };

struct RecordView {
  uint64_t counter = 0;
  uint32_t kernelIndex = UINT32_MAX;
  RecordState state = RecordState::Free;
  uint32_t expectedBodies = 0;
  uint32_t consumedBodies = 0;
};

class RecordPool {
 public:
  explicit RecordPool(size_t capacity) : slots_(capacity) {
    for (size_t i = 0; i < slots_.size(); ++i) slots_[i].view.kernelIndex = static_cast<uint32_t>(i);
  }
  uint64_t nextCounter() {
    std::lock_guard<std::mutex> lock(mutex_);
    return nextCounter_++;
  }
  ReserveResult reserve(uint64_t counter, uint32_t /*channels*/, uint32_t* kernelIndex) {
    if (!kernelIndex) return ReserveResult::ResourceExhausted;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t selected = slots_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < slots_.size(); ++i) {
      const Slot& slot = slots_[i];
      if (slot.view.state == RecordState::Free && slot.lastUse <= oldest) {
        selected = i;
        oldest = slot.lastUse;
      }
    }
    if (selected == slots_.size()) return ReserveResult::ResourceExhausted;
    Slot& slot = slots_[selected];
    slot.view.counter = counter;
    slot.view.expectedBodies = slot.view.consumedBodies = 0;
    slot.view.state = RecordState::Reserved;
    slot.bodyChannels.clear();
    slot.lastUse = ++clock_;
    *kernelIndex = static_cast<uint32_t>(selected);
    return ReserveResult::Reserved;
  }
  bool markRunning(uint32_t index) { return transition(index, RecordState::Reserved, RecordState::Running); }
  bool registerBody(uint64_t counter, uint32_t index, uint32_t channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || slot->view.counter != counter || !active(*slot)) return false;
    if (channel >= slot->bodyChannels.size()) slot->bodyChannels.resize(channel + 1, false);
    if (!slot->bodyChannels[channel]) {
      slot->bodyChannels[channel] = true;
      ++slot->view.expectedBodies;
    }
    return true;
  }
  bool consumeBody(uint64_t counter, uint32_t index, uint32_t bodyCount = 1) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || slot->view.counter != counter || !active(*slot) ||
        !slot->view.expectedBodies || !bodyCount ||
        slot->view.consumedBodies >= slot->view.expectedBodies) return false;
    const uint32_t remaining = slot->view.expectedBodies - slot->view.consumedBodies;
    slot->view.consumedBodies += bodyCount < remaining ? bodyCount : remaining;
    if (slot->view.consumedBodies == slot->view.expectedBodies) releaseLocked(*slot);
    return true;
  }
  bool view(uint32_t index, RecordView* out) const {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const Slot* slot = get(index);
    if (!slot) return false;
    *out = slot->view;
    return true;
  }
  bool find(uint64_t counter, uint32_t* kernelIndex, RecordView* out = nullptr) const {
    if (!kernelIndex) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Slot& slot : slots_) if (slot.view.state != RecordState::Free && slot.view.counter == counter) {
      *kernelIndex = slot.view.kernelIndex;
      if (out) *out = slot.view;
      return true;
    }
    return false;
  }
  size_t capacity() const { return slots_.size(); }

 private:
  struct Slot { RecordView view; std::vector<bool> bodyChannels; uint64_t lastUse = 0; };
  static bool active(const Slot& slot) {
    return slot.view.state == RecordState::Reserved || slot.view.state == RecordState::Running;
  }
  Slot* get(uint32_t index) { return index < slots_.size() ? &slots_[index] : nullptr; }
  const Slot* get(uint32_t index) const { return index < slots_.size() ? &slots_[index] : nullptr; }
  bool transition(uint32_t index, RecordState from, RecordState to) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || slot->view.state != from) return false;
    slot->view.state = to;
    return true;
  }
  void releaseLocked(Slot& slot) {
    slot.view.state = RecordState::Free;
    slot.view.counter = 0;
    slot.view.expectedBodies = slot.view.consumedBodies = 0;
    slot.bodyChannels.clear();
    slot.lastUse = ++clock_;
  }
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  uint64_t clock_ = 0;
  uint64_t nextCounter_ = 0;
};

}  // namespace rccl_ccld

#endif
