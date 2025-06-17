#include "cache_model.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

using namespace GCoM;
using namespace std;
namespace fs = std::filesystem;

// For Accel-sim cache Mode
CacheModel::CacheModel(ECacheModelType type, fs::path cacheStatPath, std::streamoff cacheStatPos, unsigned kernelIdx, std::streamoff &nextKernelCacheStatPos)
{
    mType = type;
    assert(type == ECacheModelType::ACCELSIM);

    ifstream ifs(cacheStatPath, ios::binary);
    assert(ifs.is_open() == true);
    
    // find the cache statistic of kernelIdx in the binary file
    ifs.seekg(0, ios::end);
    streamoff fileSize = ifs.tellg();
    assert(cacheStatPos < fileSize);
    ifs.seekg(cacheStatPos, ios::beg);

    struct CacheStatHeader header;
    boost::archive::binary_iarchive ia(ifs);
    ia & header;
    assert(header.kernelNumber == (int) kernelIdx);
    ia & mKernelCacheStat;

    nextKernelCacheStatPos = ifs.tellg();
}

// For Accel-sim cache Mode
// read cache statistics of Accel-sim
int CacheModel::cacheAccess(WarpInst &inst, unsigned  warpIdx, unsigned  instIdx)
{
    if (inst.mDecoded.op == EUArchOp::LOAD_OP || inst.mDecoded.op == EUArchOp::STORE_OP)
    {
        WarpInstCacheStat *matchedPtr;
        // for backward compatibility of WarpInstCacheStat version 0
        if (instIdx < mKernelCacheStat[warpIdx].size()
                && mKernelCacheStat[warpIdx][instIdx].warpInstIdx == (unsigned) -1)
        {
            matchedPtr = &mKernelCacheStat[warpIdx][instIdx];
        }
        else
        {
            // WarpInstCacheStat version 1 file only has LD/ST instructions
            for (unsigned i = 0; i < mKernelCacheStat[warpIdx].size(); i++)
            {
                if (mKernelCacheStat[warpIdx][i].warpInstIdx == instIdx)
                {
                    matchedPtr = &mKernelCacheStat[warpIdx][i];
                    break;
                }
            }
        }

        if (inst.mDecoded.pc != (GCoM::Address) (*matchedPtr).pc ||
                inst.mDecoded.op != (*matchedPtr).op)
        {
            cout << "[Error] Accel-sim Cache model: instruction mismatch" << endl;
            return -1;
        }

        inst.mMemStat = (*matchedPtr);
        for (Address addr : (*matchedPtr).accessQAddr)
        {
            MemAccess psuedoMemAccess;
            psuedoMemAccess.mAddr = addr;
            inst.mDecoded.accessQ.push_back(psuedoMemAccess);
        }

        // Collect warp instruction cache statistics
        if (inst.mDecoded.space == EMemorySpace::GLOBAL_SPACE || inst.mDecoded.space == EMemorySpace::LOCAL_SPACE)
        {
            auto key = make_pair(inst.mDecoded.pc, instIdx);
            // make element in mWarpInstStatistics_PCInstIdx2CacheHitMiss if not exist, else reference it
            Warp::WarpInstCacheHitMissCount &warpInstCStat = mWarpInstStatistics_PCInstIdx2CacheHitMiss[key];

            if (inst.mMemStat.l1Miss > 0)
                warpInstCStat.l1MissWarp += 1;
            else
                warpInstCStat.l1HitWarp += 1;

            if (inst.mMemStat.l2Miss > 0)
                warpInstCStat.l2MissWarp += 1;
            else
                warpInstCStat.l2HitWarp += 1;
        }
    }
    
    return 0;
}

int CacheModel::updateGlobalCacheStat(Warp &warp)
{
    for (unsigned instIdx = 0; instIdx < warp.mInsts.size(); instIdx++)
    {
        WarpInst &warpInst = warp.mInsts[instIdx];

        // find corresponding global cache statistics of a warp instruction
        // if not found, set to 0
        auto key = make_pair(warpInst.mDecoded.pc, instIdx);
        auto it = mWarpInstStatistics_PCInstIdx2CacheHitMiss.find(key);
        if (it != mWarpInstStatistics_PCInstIdx2CacheHitMiss.end())
            warp.mGlobalCacheStat.push_back(it->second);
        else
            warp.mGlobalCacheStat.push_back(Warp::WarpInstCacheHitMissCount());
    }
    return 0;
}

// Copide from Accel-sim hashing.cc
unsigned HashAdressWithIpolyFunction(Address higherBits, unsigned index, unsigned nBins)
{
    /*
     * Set Indexing function from "Pseudo-randomly interleaved memory."
     * Rau, B. R et al.
     * ISCA 1991
     * http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
     *
     * equations are corresponding to IPOLY(37) and are adopted from:
     * "Sacat: streaming-aware conflict-avoiding thrashing-resistant gpgpu
     * cache management scheme." Khairy et al. IEEE TPDS 2017.
     *
     * equations for 16 banks are corresponding to IPOLY(5)
     * equations for 32 banks are corresponding to IPOLY(37)
     * equations for 64 banks are corresponding to IPOLY(67)
     * To see all the IPOLY equations for all the degrees, see
     * http://wireless-systems.ece.gatech.edu/6604/handouts/Peterson's%20Table.pdf
     *
     * We generate these equations using GF(2) arithmetic:
     * http://www.ee.unb.ca/cgi-bin/tervo/calc.pl?num=&den=&f=d&e=1&m=1
     *
     * We go through all the strides 128 (10000000), 256 (100000000),...  and
     * do modular arithmetic in GF(2) Then, we create the H-matrix and group
     * each bit together, for more info read the ISCA 1991 paper
     *
     * IPOLY hashing guarantees conflict-free for all 2^n strides which widely
     * exit in GPGPU applications and also show good performance for other
     * strides.
     */
    if (nBins == 16)
    {
        std::bitset<64> a(higherBits);
        std::bitset<4> b(index);
        std::bitset<4> newIndex(index);

        newIndex[0] =
            a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[4] ^ a[3] ^ a[0] ^ b[0];
        newIndex[1] =
            a[12] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ a[1] ^ a[0] ^ b[1];
        newIndex[2] = a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[2] ^ a[1] ^ b[2];
        newIndex[3] = a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[3] ^ a[2] ^ b[3];

        return newIndex.to_ulong();
    }
    else if (nBins == 32)
    {
        std::bitset<64> a(higherBits);
        std::bitset<5> b(index);
        std::bitset<5> newIndex(index);

        newIndex[0] =
            a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[5] ^ a[3] ^ a[0] ^ b[0];
        newIndex[1] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[6] ^ a[4] ^
                       a[1] ^ b[1];
        newIndex[2] =
            a[14] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[2] ^ a[0] ^ b[2];
        newIndex[3] =
            a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[3] ^ a[1] ^ b[3];
        newIndex[4] =
            a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ a[2] ^ b[4];
        return newIndex.to_ulong();
    }
    else if (nBins == 64)
    {
        std::bitset<64> a(higherBits);
        std::bitset<6> b(index);
        std::bitset<6> newIndex(index);

        newIndex[0] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[12] ^ a[10] ^ a[6] ^ a[5] ^
                       a[0] ^ b[0];
        newIndex[1] = a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[5] ^ a[1] ^
                       a[0] ^ b[1];
        newIndex[2] = a[16] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[6] ^ a[2] ^
                       a[1] ^ b[2];
        newIndex[3] = a[17] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[7] ^ a[3] ^
                       a[2] ^ b[3];
        newIndex[4] = a[18] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[10] ^ a[8] ^ a[4] ^
                       a[3] ^ b[4];
        newIndex[5] =
            a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[11] ^ a[9] ^ a[5] ^ a[4] ^ b[5];
        return newIndex.to_ulong();
    }
    else
    { /* Else incorrect number of bins for the hashing function */
        assert(
            "\nmemory_partition_indexing error: The number of "
            "bins should be "
            "16, 32 or 64 for the hashing IPOLY index function. other bin "
            "numbers are not supported. Generate it by yourself! \n" &&
            0);

        return 0;
    }
}

// int CacheModel::setSharedMemSize(KernelInfo kernelInfo, HWConfig hwConfig)
// {
//     // Adaoptive cache adjust shared memory size and L1 D cache size dynamically
// }

// Copide from Accel-sim cache_config::hash_function
unsigned GCoM::HashAddress(Address addr, unsigned nBins, unsigned offSetBits, EHashFunction hashFunctionType)
{
    unsigned log2nBins = LogB2(nBins);

    unsigned binIndex = 0;

    switch (hashFunctionType)
    {
    case EHashFunction::FERMI_HASH_SET_FUNCTION:
    {
        /*
         * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
         * Distance Theory" Cedric Nugteren et al. HPCA 2014
         */
        unsigned lowerXor = 0;
        unsigned upperXor = 0;

        if (nBins == 32 || nBins == 64)
        {
            // Lower xor value is bits 7-11
            lowerXor = (addr >> offSetBits) & 0x1F;

            // Upper xor value is bits 13, 14, 15, 17, and 19
            upperXor = (addr & 0xE000) >> 13;   // Bits 13, 14, 15
            upperXor |= (addr & 0x20000) >> 14; // Bit 17
            upperXor |= (addr & 0x80000) >> 15; // Bit 19

            binIndex = (lowerXor ^ upperXor);

            // 48KB cache prepends the binIndex with bit 12
            if (nBins == 64)
                binIndex |= (addr & 0x1000) >> 7;
        }
        else
        {
            assert(false && "Incorrect number of bins for Fermi hashing function.\n");
        }
        break;
    }
    case EHashFunction::BITWISE_XORING_FUNCTION:
    {
        Address higherBits = addr >> (offSetBits + log2nBins);
        unsigned index = (addr >> offSetBits) & (nBins - 1);
        binIndex = (index) ^ (higherBits & (nBins - 1));
        break;
    }
    case EHashFunction::HASH_IPOLY_FUNCTION:
    {
        Address higherBits = addr >> (offSetBits + log2nBins);
        unsigned index = (addr >> offSetBits) & (nBins - 1);
        binIndex = HashAdressWithIpolyFunction(higherBits, index, nBins);
        break;
    }

    case EHashFunction::LINEAR_SET_FUNCTION:
    {
        binIndex = (addr >> offSetBits) & (nBins - 1);
        break;
    }

    default:
    {
        assert(false && "\nUndefined hash function.\n");
        break;
    }
    }

    assert(binIndex < nBins);

    return binIndex;
}