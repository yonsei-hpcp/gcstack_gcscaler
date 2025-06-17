#ifndef _WORKER_H
#define _WORKER_H

#include <stdlib.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include "common.h"

namespace GCoM
{
    class OptionParser
    {
    public:
        void ParseCommandLine(int argc, char **argv);
        void ShowHelp(char *argStr);

        // input: mConfigPath
        // output: hwConfig
        int ReadHWConfig(HWConfig &hwConfig);

        struct KernelStat
        {
            double totalCPI = 0;
            double baseCPI = 0;
            double comDataCPI = 0;
            double comStructCPI = 0;
            double memDataCPI = 0;
            double memStructCPI = 0;
            double syncCPI = 0;
            double controlCPI = 0;
            double emptyWarpSlotCPI = 0;
            double idleSubcoreCPI = 0;
            double idleSMCPI = 0;
        };
        // input: mKernelTargetCPIPath
        // output: kerneTargetCPIs
        int ReadKernelTargetStats(std::vector<KernelStat> &kerneTargetStats);
    
        // Command line options - general
        int mWorkType;
        bool mIsGCStackIdleDef;
        std::filesystem::path mRepWarpPath;
        std::filesystem::path mConfigPath;
        std::filesystem::path mSASSTracePath;

        // Command line options - RunGCoMWithAccelsimCacheModel
        std::filesystem::path mCacheStatPath;

        // Command line options - RunGCoMWithAccelsimCacheModelGCstackBaseRepWarpSearch
        std::filesystem::path mKernelTargetCPIPath;
        std::filesystem::path mExportRepWarpPath;
    };

    class Worker
    {
    public:
        Worker();
        int DoWork(int argc, char **argv);
        void ShowHelp(char *argStr);
    
    private:
        int (*WorkCases[20])(OptionParser opp);
    };

    int RunGCoMWithAccelsimCacheModel(OptionParser opp);
    int RunGCoMWithAccelsimCacheModelGPUMechRepWarpSelect(OptionParser opp);
    int RunGCoMWithAccelsimCacheModelCPIAssistedKmeansRepWarp(OptionParser opp);
    int RunGCoMWithAccelsimCacheModelGCstackBaseRepWarpSearch(OptionParser opp);
}

#endif