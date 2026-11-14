#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

namespace phoenix_compat {
// Bounded GPU snapshots, oldest first. DX9 retains submitted snapshots until
// their corresponding model outputs are consumed.
// One slot preserves low-memory/other-API behavior; extra slots hide capture
// fence lag and feed the shared-mapping ring without new allocation.
class CaptureQueue {
  public:
    static constexpr unsigned max_slots = 4;
    struct Entry {uint64_t serial=0,frame=0,tick=0;unsigned slot=0;bool retired=false;};
  private:
    std::array<Entry,max_slots> entries_{};
    unsigned size_=0,capacity_=1;
    unsigned submitted_slots_=0;
  public:
    void capacity(unsigned count) {
        if(size_||submitted_slots_||!count||count>entries_.size())throw std::logic_error("Invalid capture capacity change");
        capacity_=count;
    }
    unsigned size()const{return size_;}
    unsigned capacity()const{return capacity_;}
    Entry* front(){return size_?&entries_[0]:nullptr;}
    int free_slot()const {
        for(unsigned slot=0;slot<capacity_;++slot) {
            bool used=(submitted_slots_ & (1u << slot))!=0;
            for(unsigned i=0;i<size_;++i)used|=entries_[i].slot==slot;
            if(!used)return int(slot);
        }
        return -1;
    }
    void push(Entry entry) {
        if(size_==capacity_||entry.slot>=capacity_||!entry.serial||!entry.frame||
            (submitted_slots_ & (1u << entry.slot)))
            throw std::logic_error("Invalid pending capture");
        for(unsigned i=0;i<size_;++i)if(entries_[i].slot==entry.slot||entries_[i].serial>=entry.serial)
            throw std::logic_error("Capture slot or fence reused before retirement");
        entries_[size_++]=entry;
    }
    void pop(bool retain_snapshot = false) {
        if(!size_||!entries_[0].retired)throw std::logic_error("Capture released before retirement");
        if(retain_snapshot)submitted_slots_ |= 1u << entries_[0].slot;
        for(unsigned i=1;i<size_;++i)entries_[i-1]=entries_[i];
        entries_[--size_]={};
    }
    void release_snapshot(unsigned slot) {
        if(slot>=capacity_ || !(submitted_slots_ & (1u << slot)))
            throw std::logic_error("Capture snapshot was not submitted");
        submitted_slots_ &= ~(1u << slot);
    }
};
}
