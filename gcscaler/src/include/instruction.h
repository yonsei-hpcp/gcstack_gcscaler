#ifndef _INSTRUCTION_H
#define _INSTRUCTION_H

#include <vector>
#include <list>
#include <cstdint>
#include <bitset>
#include "common.h"
#include "trace_parser.h"

#define MAX_WARP_SIZE 32

namespace GCoM
{
    typedef std::bitset<MAX_WARP_SIZE> ActiveMask;

    enum class EUArchOp
    {
        NO_OP = -1,
        ALU_OP = 1,
        SFU_OP,
        TENSOR_CORE_OP,
        DP_OP,
        SP_OP,
        INTP_OP,
        ALU_SFU_OP,
        LOAD_OP,
        TENSOR_CORE_LOAD_OP,
        TENSOR_CORE_STORE_OP,
        STORE_OP,
        BRANCH_OP,
        BARRIER_OP,
        MEMORY_BARRIER_OP,
        CALL_OPS,
        RET_OPS,
        EXIT_OPS,
        CONST_MEM_OP, // added to count inst mix
        SHARED_MEM_OP, // added to count inst mix
        TEX_MEM_OP,// added to count inst mix
        ETC_MEM_OP,// added to count inst mix
        LAST_NORM_OP,// added to count inst mix
        SPECIALIZED_UNIT_1_OP,
        SPECIALIZED_UNIT_2_OP,
        SPECIALIZED_UNIT_3_OP,
        SPECIALIZED_UNIT_4_OP,
        SPECIALIZED_UNIT_5_OP,
        SPECIALIZED_UNIT_6_OP,
        SPECIALIZED_UNIT_7_OP,
        SPECIALIZED_UNIT_8_OP,
        LAST_SPEC_OP // added to count inst mix
    };

    enum class EMemorySpace
    {
        UNDEFINED_SPACE = 0,
        GLOBAL_SPACE,
        LOCAL_SPACE,
        SHARED_SPACE,
        CONST_SPACE,
        TEX_SPACE,
        INSTRUCTION_SPACE
    };

    enum class ECacheOpType
    {
        CACHE_UNDEFINED,

        // loads
        CACHE_ALL,       // .ca
        CACHE_LAST_USE,  // .lu
        CACHE_VOLATILE,  // .cv
        CACHE_L1,        // .nc

        // loads and stores
        CACHE_STREAMING,  // .cs
        CACHE_GLOBAL,     // .cg

        // stores
        CACHE_WRITE_BACK,    // .wb
        CACHE_WRITE_THROUGH  // .wt
    };

    enum class EMemAccessType
    {
        GLOBAL_ACC_R,
        LOCAL_ACC_R,
        TEXTURE_ACC_R,
        GLOBAL_ACC_W,
        LOCAL_ACC_W,
        TEXTURE_ACC_w,
        L1_WRBK_ACC,
        L2_WRBK_ACC,
        INST_ACC_R,
        L1_WR_ALLOC_R,
        L2_WR_ALLOC_R
    };

    enum class EOpLatencyInitIntvType
    {
        DEFALUT,
        INT,
        FP,
        FP16,
        DP,
        SFU,
        TENSOR,
        SPECIALIZED_UNIT_1,
        SPECIALIZED_UNIT_2,
        SPECIALIZED_UNIT_3,
        SPECIALIZED_UNIT_4,
        SPECIALIZED_UNIT_5,
        SPECIALIZED_UNIT_6,
        SPECIALIZED_UNIT_7,
        SPECIALIZED_UNIT_8
    };

    class MemAccess
    {
    public:
        Address mAddr;
        bool mbIsWrite;
        unsigned mReqSize; // bytes
        EMemAccessType mType;
    };

    class Warp; // Forward declaration

    class WarpInst
    {
    public:
        WarpInst() 
        {
            //mWarpPtr = warpPtr;
        }
        ~WarpInst() {}

        //unsigned mInstsId = 3;

        unsigned mDataSize; // (B) need for each thread
        std::vector<Address> mPerThreadMemAddr; // memory address for each thread
        EOpLatencyInitIntvType mOpLatencyInitIntvType; // operation latency and initiation interval type
        struct 
        {
            Address pc = (Address) -1;
            EUArchOp op = EUArchOp::NO_OP;
            std::vector<unsigned> out; // dst register indexs
            std::vector<unsigned> in; // src register indexs
            int pred; // predicate register index
            unsigned latency = 0; // operation latency
            unsigned initiationInterval = 0;
            EMemorySpace space = EMemorySpace::UNDEFINED_SPACE;
            ECacheOpType cacheOp = ECacheOpType::CACHE_UNDEFINED;
            ActiveMask activeMask; // dynamic active mask for timing model (after predication)
            std::list<MemAccess> accessQ; // memory accesses after coalescing

        } mDecoded;

        struct MemStat
        {
            uint8_t l1Hit = 0; // # of thread memory request that hit L1
            uint8_t l1Miss = 0; // # of all thread memory request that missed L1
            uint8_t coalescedL1Miss = 0; // L1 <-> L2 mem request #. Miss on the same MSHR entry will be coalesced.
            uint8_t l2Hit = 0; // # of L1 <-> L2 mem request that hit L2
            uint8_t l2Miss = 0; // # of L1 <-> L2 mem request that missed L2

            MemStat& operator=(const struct WarpInstCacheStat& other);
        } mMemStat;

        //inst_trace_t *mInstTracePtr; // trace before decoding // not necessary?
        Warp *mWarpPtr; // backward pointer to warp class
    };

    class Warp
    {
    public:
        Warp() {}
        ~Warp() {}

        // custom copy constructor
        Warp(const Warp &other)
        {
            mInsts = other.mInsts;
            for (WarpInst &inst : mInsts)
                inst.mWarpPtr = this;
            mGlobalCacheStat = other.mGlobalCacheStat;
        }

        std::vector<WarpInst> mInsts;
        // struct
        // {
        //     std::string kernelName;
        //     unsigned ctaId; // CTA id among all CTAs in a kernel // not neccessary? 
        //     unsigned warpId; // warp id among warps in a CTA // not neccessary?
        // } mInfo;

        struct WarpInstCacheHitMissCount
        {
            unsigned l1HitWarp = 0; // The number of warp instructions that did not miss L1 cache at all
            unsigned l1MissWarp = 0; //The number of warp instructions that missed L1
            unsigned l2HitWarp = 0; // The number of warp instructions that did not miss L2 cache at all
            unsigned l2MissWarp = 0; //The number of warp instructions that missed L2
        };
        std::vector<WarpInstCacheHitMissCount> mGlobalCacheStat;
    };
}

#endif