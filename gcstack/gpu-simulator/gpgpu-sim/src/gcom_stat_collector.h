#ifndef __GCOM_STAT_COLLECTOR_H__
#define __GCOM_STAT_COLLECTOR_H__

#include <string>
#include <vector>
#include "abstract_hardware_model.h"


class GComStatCollector {
  public:
  GComStatCollector(gpgpu_context *ctx);
  ~GComStatCollector() {}

  // input: *kernelInfo
  // output: initialized mKernelCacheStat 
  void BeginKernel(kernel_info_t *kernelInfo);

  // input: warpInst, ctaId, dynamicWarpId
  // output: updated mKernelCacheStat[warpIdx].{pc, op, issueCycle, accessQAddr}
  void AddInstAccessq(warp_inst_t &warpInst, unsigned ctaId, unsigned dynamicWarpId);

  enum class EMemStatus {
    none,
    l1Miss,
    l1Hit,
    l2Miss,
    l2Hit,
    l1HitReserved
  };
  // input: warpInst, ctaId, dynamicWarpId, status
  // output: updated mKernelCacheStat[warpIdx].{l1Hit, l1Miss, coalescedL1Miss, l2Hit, l2Miss}
  void StatMemAccess(const warp_inst_t &warpInst, unsigned ctaId, unsigned dynamicWarpId, EMemStatus status);

  // input: mCtx->...->{launch_name, gpu_tot_sim_cycle, gpu_tot_sim_insn}, mKernelCacheStat
  // output: updated mKernelNumber, mPrevTotCycle, mPrevThreadInsn, output files(accelsim_cache_stat_*.bin kernel_cpi_*.bin)
  void EndKernel();

  struct WarpInstCacheStat 
  {
    int pc;
    int op;
    unsigned warpInstIdx;
    unsigned long long issueCycle;
    uint8_t l1Hit = 0;
    uint8_t l1Miss = 0;
    uint8_t coalescedL1Miss = 0;  // Miss to the same MSHR entry is coalesced
    uint8_t l2Hit = 0;
    uint8_t l2Miss = 0;
    std::vector<new_addr_type> accessQAddr;
  };

  struct KernelStat
  {
    double totalCPI;
    double baseCPI;
    double comDataCPI;
    double comStructCPI;
    double memDataCPI;
    double memStructCPI;
    double syncCPI;
    double controlCPI;
    double emptyWarpSlotCPI;
    double idleSubcoreCPI;
    double idleSMCPI;
  };

  struct PrevKernelStats
  {
    // Accumulated stats at the end of the previous kernel
    unsigned long long totCycle = 0;
    unsigned long long threadInsn = 0;
    double baseCycle = 0;
    double comDataCycle = 0;
    double comStructCycle = 0;
    double memDataCycle = 0;
    double memStructCycle = 0;
    double syncCycle = 0;
    double controlCycle = 0;
    double emptyWarpSlotCycle = 0;
    double subcoreIdleCycle = 0;
    double smIdleCycle = 0;
  };

  private:
  gpgpu_context *mCtx;

  std::vector<std::vector<WarpInstCacheStat> > mKernelCacheStat;
  std::vector<unsigned> mWarpInstHeadIdx;
  std::vector<KernelStat> mKernelStats;
  int mNumWarpPerCTA;
  int mKernelNumber = 0;
  PrevKernelStats mPrevKernelStats;
  std::string mOutputFileIdentifier = "";

  // input: mKernelStats, mOutputFileIdentifier, mKernelCacheStat
  // output: updated output files(accelsim_cache_stat_*.bin kernel_cpi_*.bin)
  void ExportKernelGCoMStat();

};

#endif