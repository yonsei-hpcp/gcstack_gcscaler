#ifndef _CACHE_MODEL_H
#define _CACHE_MODEL_H

#include <map>
#include <vector>
#include <filesystem>
#include "instruction.h"
#include <string>
#include <vector>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace fs = std::filesystem;

namespace GCoM
{
    enum class ECacheModelType
    {
        ACCELSIM // Read compressed cache stat came out from reference
    };

    // Data structure shared with Accel-sim 
    struct WarpInstCacheStat
    {
        int pc;
        EUArchOp op;
        unsigned warpInstIdx;
        uint8_t l1Hit; // # of thread memory request that hit L1
        uint8_t l1Miss; // # of all thread memory request that missed L1
        uint8_t coalescedL1Miss; // L1 <-> L2 mem request #. Miss on the same MSHR entry will be coalesced.
        uint8_t l2Hit; // # of L1 <-> L2 mem request that hit L2
        uint8_t l2Miss; // # of L1 <-> L2 mem request that missed L2
        std::vector<Address> accessQAddr; // memory access addresses after coalescing

        template<class Archive>
        void serialize(Archive &ar, const unsigned version)
        {
            unsigned rawOp = static_cast<unsigned>(op);
            ar & pc;
            ar & rawOp;
            op = static_cast<EUArchOp>(rawOp);
            if (version == 1)
            {
                ar & warpInstIdx;
                ar & l1Hit;
                ar & l1Miss;
                ar & coalescedL1Miss;
                ar & l2Hit;
                ar & l2Miss;
            } else if (version == 0)
            {
                unsigned v0L1Hit, v0L1Miss, v0CoalescedL1Miss, v0L2Hit, v0L2Miss;
                warpInstIdx = -1;
                ar & v0L1Hit;
                ar & v0L1Miss;
                ar & v0CoalescedL1Miss;
                ar & v0L2Hit;
                ar & v0L2Miss;
                l1Hit = (uint8_t) v0L1Hit;
                l1Miss = (uint8_t) v0L1Miss;
                coalescedL1Miss = (uint8_t) v0CoalescedL1Miss;
                l2Hit = (uint8_t) v0L2Hit;
                l2Miss = (uint8_t) v0L2Miss;
            }
            ar & accessQAddr;
        }
    };

    // Data structure shared with Accel-sim 
    struct CacheStatHeader
    {
        int kernelNumber;
        std::streamoff nextKernelOffset; // Deprecated. Leave for binary file compatibility.

        template<class Archive>
        void serialize(Archive &ar, const unsigned  version)
        {
            ar & kernelNumber;
            ar & nextKernelOffset;
        }
    };

    // warper class for cache model
    class CacheModel
    {
    public:
        CacheModel() {}
        CacheModel(ECacheModelType type)
        {
            mType = type;
        }

        // Initialization for Accel-sim cache mode
        // Output: nextKernelCacheStatPos
        CacheModel(ECacheModelType type, fs::path cacheStatPath, std::streamoff cacheStatPos, unsigned  kernelIdx, std::streamoff &nextKernelCacheStatPos);

        /*
        * Access cache and set mMemStat of WarpInst
        * return 0 for normal exit. -1 for abnormal exit.
        */
        // for Accel-sim cache mode
        int cacheAccess(WarpInst &inst, unsigned  warpIdx, unsigned  instIdx); 

        // update global cache statistics of a warp with collected warp instruction statistics
        int updateGlobalCacheStat(Warp &warp);
        
    private:
        ECacheModelType mType;

        // do memory coalescing and generate MemAccess
        // GenerateMemAccesses(WarpInst &inst)

        // warp instruction statistic collected during cahce access
        std::map<std::pair<Address, unsigned>, Warp::WarpInstCacheHitMissCount>
                mWarpInstStatistics_PCInstIdx2CacheHitMiss;

        // Accel-sim cache mode only
        // to hold kernel cache statistics from accel-sim
        std::vector<std::vector<WarpInstCacheStat> > mKernelCacheStat;

        // based on shader_core_config::max_cta of Accel-sim
        // input: kernelInfo, hwConfig
        // output: 
        // int setSharedMemSize(KernelInfo kernelInfo, HWConfig hwConfig);
    };

    // Copide from Accel-sim cache_config::hash_function
    // Currently bank index hashing is used in a warp profiling. Otherwise this may go inside CacheModel.
    // L1, L2 set, bank index hashing function
    unsigned HashAddress(Address addr, unsigned nBins, unsigned offSetBits, EHashFunction hashFunctionType);
} // namespace GCoM

BOOST_CLASS_VERSION(GCoM::WarpInstCacheStat, 1)

#endif