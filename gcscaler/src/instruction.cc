#include "instruction.h"
#include "cache_model.h"

GCoM::WarpInst::MemStat& GCoM::WarpInst::MemStat::operator=(const struct WarpInstCacheStat& other)
{
    l1Hit = other.l1Hit;
    l1Miss = other.l1Miss;
    coalescedL1Miss = other.coalescedL1Miss;
    l2Hit = other.l2Hit;
    l2Miss = other.l2Miss;
    return *this;
}