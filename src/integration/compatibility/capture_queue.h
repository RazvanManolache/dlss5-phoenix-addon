#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

namespace phoenix_compat {
// Bounded GPU snapshots, oldest first. A retired snapshot remains leased until
// submission: its DX9 texture is also the source paired with the model output.
// One slot preserves low-memory/other-API behavior; two hide capture fence lag.
class CaptureQueue {
  public:
    struct Entry {uint64_t serial=0,frame=0,tick=0;unsigned slot=0;bool retired=false;};
  private:
    std::array<Entry,2> entries_{};
    unsigned size_=0,capacity_=1;
  public:
    void capacity(unsigned count) {
        if(size_||!count||count>entries_.size())throw std::logic_error("Invalid capture capacity change");
        capacity_=count;
    }
    unsigned size()const{return size_;}
    unsigned capacity()const{return capacity_;}
    Entry* front(){return size_?&entries_[0]:nullptr;}
    int free_slot()const {
        for(unsigned slot=0;slot<capacity_;++slot) {
            bool used=false;for(unsigned i=0;i<size_;++i)used|=entries_[i].slot==slot;
            if(!used)return int(slot);
        }
        return -1;
    }
    void push(Entry entry) {
        if(size_==capacity_||entry.slot>=capacity_||!entry.serial||!entry.frame)
            throw std::logic_error("Invalid pending capture");
        for(unsigned i=0;i<size_;++i)if(entries_[i].slot==entry.slot||entries_[i].serial>=entry.serial)
            throw std::logic_error("Capture slot or fence reused before retirement");
        entries_[size_++]=entry;
    }
    void pop() {
        if(!size_||!entries_[0].retired)throw std::logic_error("Capture released before retirement");
        if(size_==2)entries_[0]=entries_[1];
        entries_[--size_]={};
    }
};
}
