#ifndef RCCL_CCLD_RECORD_POOL_H_
#define RCCL_CCLD_RECORD_POOL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace rccl_ccld {

enum class RecordState : uint8_t {
  Free,
  Reserved,
  Running,
  Complete,
  Consumed,
  ResourceExhausted,
  ObservationDropped,
  MissingCompletion
};

enum class ReserveResult : uint8_t { Reserved, ResourceExhausted };

struct RecordView {
  uint64_t counter = 0;
  uint32_t kernelIndex = UINT32_MAX;
  RecordState state = RecordState::Free;
  uint32_t expectedChannels = 0;
  uint32_t completedChannels = 0;
};

// Host-side lifecycle model for pinned diagnostic bodies. Device storage is
// deliberately separate; kernelIndex is only the physical body index.
class RecordPool {
 public:
  explicit RecordPool(size_t capacity) : slots_(capacity) {
    for (size_t i = 0; i < slots_.size(); ++i) slots_[i].view.kernelIndex = static_cast<uint32_t>(i);
  }

  uint64_t nextCounter() {
    std::lock_guard<std::mutex> lock(mutex_);
    return nextCounter_++;
  }

  ReserveResult reserve(uint64_t counter, uint32_t channels, uint32_t* kernelIndex) {
    if (channels == 0 || kernelIndex == nullptr) return ReserveResult::ResourceExhausted;
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
    slot.view.expectedChannels = channels;
    slot.view.completedChannels = 0;
    slot.view.state = RecordState::Reserved;
    slot.channelDone.assign(channels, false);
    slot.channelRegistered.assign(channels, false);
    slot.expectedPlans = 0;
    slot.completedPlans = 0;
    slot.lastUse = ++clock_;
    *kernelIndex = static_cast<uint32_t>(selected);
    return ReserveResult::Reserved;
  }

  bool markRunning(uint32_t index) {
    return transition(index, RecordState::Reserved, RecordState::Running);
  }

  bool markChannelComplete(uint32_t index, uint32_t channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || slot->view.state != RecordState::Running || channel >= slot->channelDone.size() || slot->channelDone[channel]) return false;
    slot->channelDone[channel] = true;
    ++slot->view.completedChannels;
    return true;
  }

  bool registerChannel(uint32_t index, uint32_t channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || (slot->view.state != RecordState::Reserved && slot->view.state != RecordState::Running)) return false;
    if (channel >= slot->channelDone.size()) {
      slot->channelDone.resize(channel + 1, false);
      slot->channelRegistered.resize(channel + 1, false);
    }
    if (!slot->channelRegistered[channel]) {
      slot->channelRegistered[channel] = true;
      ++slot->view.expectedChannels;
    }
    return true;
  }

  bool expectPlan(uint32_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || (slot->view.state != RecordState::Reserved && slot->view.state != RecordState::Running)) return false;
    ++slot->expectedPlans;
    return true;
  }

  bool completePlan(uint32_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || (slot->view.state != RecordState::Running && slot->view.state != RecordState::Reserved) ||
        slot->completedPlans >= slot->expectedPlans || slot->expectedPlans == 0) return false;
    ++slot->completedPlans;
    if (slot->completedPlans == slot->expectedPlans) slot->view.state = RecordState::Complete;
    return slot->view.state == RecordState::Complete;
  }

  bool markPlanComplete(uint32_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || slot->view.state != RecordState::Running ||
        slot->view.completedChannels != slot->view.expectedChannels) return false;
    slot->view.state = RecordState::Complete;
    return true;
  }

  bool markMissingCompletion(uint32_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || (slot->view.state != RecordState::Reserved && slot->view.state != RecordState::Running)) return false;
    slot->view.state = RecordState::MissingCompletion;
    return true;
  }

  bool markConsumed(uint32_t index) {
    return transition(index, RecordState::Complete, RecordState::Consumed);
  }

  bool release(uint32_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || (slot->view.state != RecordState::Consumed &&
                  slot->view.state != RecordState::MissingCompletion &&
                  slot->view.state != RecordState::ObservationDropped &&
                  slot->view.state != RecordState::ResourceExhausted)) return false;
    slot->view.state = RecordState::Free;
    slot->view.counter = 0;
    slot->view.expectedChannels = 0;
    slot->view.completedChannels = 0;
    slot->channelDone.clear();
    slot->channelRegistered.clear();
    slot->expectedPlans = 0;
    slot->completedPlans = 0;
    slot->lastUse = ++clock_;
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

  size_t capacity() const { return slots_.size(); }

 private:
  struct Slot {
    RecordView view;
    std::vector<bool> channelDone;
    std::vector<bool> channelRegistered;
    uint64_t lastUse = 0;
    uint32_t expectedPlans = 0;
    uint32_t completedPlans = 0;
  };

  Slot* get(uint32_t index) {
    return index < slots_.size() ? &slots_[index] : nullptr;
  }
  const Slot* get(uint32_t index) const {
    return index < slots_.size() ? &slots_[index] : nullptr;
  }
  bool transition(uint32_t index, RecordState from, RecordState to) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot* slot = get(index);
    if (!slot || slot->view.state != from) return false;
    slot->view.state = to;
    return true;
  }

  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  uint64_t clock_ = 0;
  uint64_t nextCounter_ = 0;
};

}  // namespace rccl_ccld

#endif
