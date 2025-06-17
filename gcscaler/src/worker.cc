#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include "common.h"
#include "worker.h"

using namespace GCoM;
using namespace std;

GCoM::Worker::Worker()
{
    WorkCases[0] = RunGCoMWithAccelsimCacheModel;
    WorkCases[2] = RunGCoMWithAccelsimCacheModelGPUMechRepWarpSelect;
    WorkCases[3] = RunGCoMWithAccelsimCacheModelCPIAssistedKmeansRepWarp;
    WorkCases[4] = RunGCoMWithAccelsimCacheModelGCstackBaseRepWarpSearch;
}

int GCoM::Worker::DoWork(int argc, char **argv)
{
    OptionParser opp;
    opp.ParseCommandLine(argc, argv);

    return WorkCases[opp.mWorkType](opp);
}

void GCoM::OptionParser::ParseCommandLine(int argc, char **argv)
{
    if (argc < 2)
    {
        ShowHelp(argv[0]);
        exit(-1);
    }

    mIsGCStackIdleDef = true; // use GCStack idle definition which accounts all SMs in default
    char opt;
    while ((opt = getopt(argc, argv, "hw:t:c:r:C:k:o:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            ShowHelp(argv[0]);
            exit(0);
        case 'w':
            mWorkType = atoi(optarg);
            break;
        case 't':
            mSASSTracePath = optarg;
            break;
        case 'c':
            mCacheStatPath = optarg;
            break;
        case 'r':
            mRepWarpPath = optarg;
            break;
        case 'C':
            mConfigPath = optarg;
            break;
        case 'k':
            mKernelTargetCPIPath = optarg;
            break;
        case 'a':
            mIsGCStackIdleDef = false;
            break;
        case 'o':
            mExportRepWarpPath = optarg;
            break;
        }
    }
}

void GCoM::OptionParser::ShowHelp(char *argStr)
{
    printf(
        "usage : %s \n"
        " -h    help\n"
        " -w [INT] work ID\n"
        " -C [path] configuration file path\n"
        " -t [path] benchmark trace directory\n"
        " -a (Optional) Use GCoM idle definition which only accounts active SMs\n"
        "\n"
        "[work 0] RunGCoMWithAccelsimCacheModel options\n"
        " -c [path] cache statistics trace path\n"
        " -r [path] representative warp index trace path\n"
        "\n"
        "[work 2] RunGCoMWithAccelsimCacheModelGPUMechRepWarpSelect options\n"
        " -c [path] cache statistics trace path\n"
        "\n"
        "[work 3] RunGCoMWithAccelsimCacheModelCPIAssistedKmeansRepWarp options\n"
        " -c [path] cache statistics trace path\n"
        " -r [path] representative warp index trace path\n"
        "\n"
        "[work 4] RunGCoMWithAccelsimCacheModelGCstackBaseRepWarpSearch options\n"
        " -c [path] cache statistics trace path\n"
        " -k [path] kernel target CPI Stack file path for representative warp search\n"
        " -o [path] (Optional) export selected representative warp index to [path]\n"
        ,
        argStr
    );
}

int GCoM::OptionParser::ReadHWConfig(HWConfig &hwConfig)
{
    ifstream ifs(mConfigPath);
    if (!ifs.is_open())
    {
        cerr << "[Error] cannot open file " << mConfigPath << endl;
        cerr << "[Warning] Default HW configuration (RTX2060) is used instead" << endl;
        return -1;
    }

    
    string line;
    while (getline(ifs, line))
    {
        // Skip comments
        size_t commentPos = line.find("//");
        if (commentPos != string::npos)
            line = line.substr(0, commentPos);

        istringstream iss(line);
        // remove leading whitespace
        while (iss.peek() == ' ' || iss.peek() == '\t')
            iss.ignore();

        string key;
        if (!getline(iss, key, ' ')) // skip type field
            continue;
        if (!getline(iss, key, '='))
            continue;

        string value;
        if (!getline(iss, value))
            continue;

        // Remove leading and trailing whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "numSMs") hwConfig.numSMs = stoul(value);
        else if (key == "numSubcorePerSM") hwConfig.numSubcorePerSM = stoul(value);
        else if (key == "maxThreadsPerSM") hwConfig.maxThreadsPerSM = stoul(value);
        else if (key == "maxCTAPerSM") hwConfig.maxCTAPerSM = stoul(value);
        else if (key == "registerPerSM") hwConfig.registerPerSM = stoul(value);
        else if (key == "coreFreq") hwConfig.coreFreq = stod(value);
        else if (key == "wrapSchedulePolicy") hwConfig.wrapSchedulePolicy = value;
        else if (key == "intLatency") hwConfig.intLatency = stoul(value);
        else if (key == "fpLatency") hwConfig.fpLatency = stoul(value);
        else if (key == "dpLatency") hwConfig.dpLatency = stoul(value);
        else if (key == "sfuLatency") hwConfig.sfuLatency = stoul(value);
        else if (key == "specializedUnit1Latency") hwConfig.specializedUnit1Latency = stoul(value);
        else if (key == "specializedUnit2Latency") hwConfig.specializedUnit2Latency = stoul(value);
        else if (key == "specializedUnit3Latency") hwConfig.specializedUnit3Latency = stoul(value);
        else if (key == "specializedUnit4Latency") hwConfig.specializedUnit4Latency = stoul(value);
        else if (key == "specializedUnit5Latency") hwConfig.specializedUnit5Latency = stoul(value);
        else if (key == "specializedUnit6Latency") hwConfig.specializedUnit6Latency = stoul(value);
        else if (key == "specializedUnit7Latency") hwConfig.specializedUnit7Latency = stoul(value);
        else if (key == "specializedUnit8Latency") hwConfig.specializedUnit8Latency = stoul(value);
        else if (key == "intInitIntv") hwConfig.intInitIntv = stoul(value);
        else if (key == "fpInitIntv") hwConfig.fpInitIntv = stoul(value);
        else if (key == "dpInitIntv") hwConfig.dpInitIntv = stoul(value);
        else if (key == "sfuInitIntv") hwConfig.sfuInitIntv = stoul(value);
        else if (key == "specializedUnit1IntiIntv") hwConfig.specializedUnit1IntiIntv = stoul(value);
        else if (key == "specializedUnit2IntiIntv") hwConfig.specializedUnit2IntiIntv = stoul(value);
        else if (key == "specializedUnit3IntiIntv") hwConfig.specializedUnit3IntiIntv = stoul(value);
        else if (key == "specializedUnit4IntiIntv") hwConfig.specializedUnit4IntiIntv = stoul(value);
        else if (key == "specializedUnit5IntiIntv") hwConfig.specializedUnit5IntiIntv = stoul(value);
        else if (key == "specializedUnit6IntiIntv") hwConfig.specializedUnit6IntiIntv = stoul(value);
        else if (key == "specializedUnit7IntiIntv") hwConfig.specializedUnit7IntiIntv = stoul(value);
        else if (key == "specializedUnit8IntiIntv") hwConfig.specializedUnit8IntiIntv = stoul(value);
        else if (key == "computePipeline") hwConfig.computePipeline = stoul(value);
        else if (key == "loadPipeline") hwConfig.loadPipeline = stoul(value);
        else if (key == "storePipeline") hwConfig.storePipeline = stoul(value);
        else if (key == "operandCollectorQue") hwConfig.operandCollectorQue = stoul(value);
        else if (key == "fuQue") hwConfig.fuQue = stoul(value);
        else if (key == "regFileReadThroughput") hwConfig.regFileReadThroughput = stoul(value);
        else if (key == "l1DLatency") hwConfig.l1DLatency = stoul(value);
        else if (key == "l1DWritePolicy") hwConfig.l1DWritePolicy = static_cast<EWritePolicy>(stoul(value));
        else if (key == "l1DWriteAllocatePolicy") hwConfig.l1DWriteAllocatePolicy = static_cast<EWriteAllocatePolicy>(stoul(value));
        else if (key == "l1DnSet") hwConfig.l1DnSet = stoul(value);
        else if (key == "l1DAssoc") hwConfig.l1DAssoc = stoul(value);
        else if (key == "l1DLineSize") hwConfig.l1DLineSize = stoul(value);
        else if (key == "l1DnBank") hwConfig.l1DnBank = stoul(value);
        else if (key == "l1DBankHashFunction") hwConfig.l1DBankHashFunction = static_cast<EHashFunction>(stoul(value));
        else if (key == "l1DLog2BankInterleaveByte") hwConfig.l1DLog2BankInterleaveByte = stoul(value);
        else if (key == "nSectorPerLine") hwConfig.nSectorPerLine = stoul(value);
        else if (key == "nMSHR") hwConfig.nMSHR = stoul(value);
        else if (key == "sharedMemLatency") hwConfig.sharedMemLatency = stoul(value);
        else if (key == "sharedMemSizeOption")
        {
            hwConfig.sharedMemSizeOption.clear();
            istringstream ss(value);
            string size;
            getline(ss, size, '{');
            while (getline(ss, size, ','))
            {
                hwConfig.sharedMemSizeOption.push_back(stoul(size));
            }
        }
        else if (key == "L2Latency") hwConfig.L2Latency = stoul(value);
        else if (key == "L2WritePolicy") hwConfig.L2WritePolicy = static_cast<EWritePolicy>(stoul(value));
        else if (key == "L2WriteAllocatePolicy") hwConfig.L2WriteAllocatePolicy = static_cast<EWriteAllocatePolicy>(stoul(value));
        else if (key == "maxNoCBW") hwConfig.maxNoCBW = stod(value);
        else if (key == "dramLatency") hwConfig.dramLatency = stoul(value);
        else if (key == "maxDRAMBW") hwConfig.maxDRAMBW = stod(value);
    }

    return 0;
}

struct KernelStatPrint
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

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) 
    {
        ar & totalCPI;
        if (version >= 1)
        {
            ar & baseCPI;
            ar & comDataCPI;
            ar & comStructCPI;
            ar & memDataCPI;
            ar & memStructCPI;
            ar & syncCPI;
            ar & controlCPI;
            ar & emptyWarpSlotCPI;
            ar & idleSubcoreCPI;
            ar & idleSMCPI;

            // bug fix of idleSMCPI calculation error in GCStack code
            idleSMCPI = totalCPI - baseCPI - comDataCPI - comStructCPI - memDataCPI - memStructCPI - syncCPI - controlCPI - emptyWarpSlotCPI - idleSubcoreCPI;
        }
    }
};
BOOST_CLASS_VERSION(KernelStatPrint, 1)

int GCoM::OptionParser::ReadKernelTargetStats(vector<KernelStat> &kerneTargetStats)
{
    // read binary file
    ifstream ifs(mKernelTargetCPIPath, ios::binary);
    if (!ifs.is_open())
    {
        cerr << "[Error] cannot open file " << mKernelTargetCPIPath << endl;
        return -1;
    }

    vector<KernelStatPrint> kernelStats;
    bool isWrongBinVersion = false;
    try 
    {
        boost::archive::binary_iarchive ia(ifs);
        ia & kernelStats;
    }
    catch (const exception &e)
    {
        isWrongBinVersion = true;
    }

    vector<double> kernelTargetCPIs;
    if (isWrongBinVersion)
    {
        try 
        {
            ifs.seekg(0);
            boost::archive::binary_iarchive ia(ifs);
            ia & kernelTargetCPIs;
        }
        catch (const exception &e)
        {
            cerr << "[Error] cannot read kernel target CPIs: " << e.what() << endl;
            return -1;
        }

        for (double cpi : kernelTargetCPIs)
        {
            KernelStat kStat;
            kStat.totalCPI = cpi;
            kerneTargetStats.push_back(kStat);
        }
    }
    else
    {
        for (KernelStatPrint kStatP : kernelStats)
        {
            KernelStat kStat;
            kStat.totalCPI = kStatP.totalCPI;
            kStat.baseCPI = kStatP.baseCPI;
            kStat.comDataCPI = kStatP.comDataCPI;
            kStat.comStructCPI = kStatP.comStructCPI;
            kStat.memDataCPI = kStatP.memDataCPI;
            kStat.memStructCPI = kStatP.memStructCPI;
            kStat.syncCPI = kStatP.syncCPI;
            kStat.controlCPI = kStatP.controlCPI;
            kStat.emptyWarpSlotCPI = kStatP.emptyWarpSlotCPI;
            kStat.idleSubcoreCPI = kStatP.idleSubcoreCPI;
            kStat.idleSMCPI = kStatP.idleSMCPI;
            kerneTargetStats.push_back(kStat);
        }
    }

    ifs.close();
    
    return 0;
}