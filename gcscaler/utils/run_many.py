#!/usr/bin/env python3

import argparse
import os
import subprocess
import tempfile
from multiprocessing import Pool
import pandas as pd
import yaml
import re
import hashlib
import time
import glob

## USER Configuration ##
APP_ARG_DICT = dict()
READ_DEFINE_ALL_APPS = True

CONFIGS = ['RTX3070',
           'RTX3070_L1_AMAT_2', 'RTX3070_L1_AMAT_50', 'RTX3070_L1_AMAT_25', # L1_AMAT_CONFIG
           'RTX3070_L2_AMAT_2', 'RTX3070_L2_AMAT_50', 'RTX3070_L2_AMAT_25', # L2_AMAT_CONFIG
           'RTX3070_D_AMAT_2', 'RTX3070_D_AMAT_50', 'RTX3070_D_AMAT_25', # D_AMAT_CONFIG
           'RTX3070_L1_CacheSize_2', 'RTX3070_L1_CacheSize_4', 'RTX3070_L1_CacheSize_50', # L1_CACHE_CONFIG
           'RTX3070_SM_23', 'RTX3070_SM_69', # SM_NUM_CONFIG
           'RTX3070_ALULat_2', 'RTX3070_ALULat_4', # ALU_LAT_CONFIG
           'RTX3070_PCom',
           'RTX3070_L1P', 'RTX3070_PMem',
           'RTX3070_Init_2', 'RTX3070_Init_4' # ALU_INIT_CONFIG
           ]


## Constants ##
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

## Function definitions ##
def get_args():
    parser = argparse.ArgumentParser(description="Executing many workloads")
    
    parser.add_argument("-w", dest="mode", type=int, default=4,help="Default=4 [2: RunGCoMWith...GPUMechRepWarpSelect, 3: Run GCoM with CPIAssistedKmeans, 4: RunGCoM With CPIBaseRepWarpSearch]")
    parser.add_argument("-C", dest="configs", type=str, nargs='+', default="all",help="HW configurations to execute [all | RTX3070 RTX3070_ALULat_2 RTX3070_ALULat_4 RTX3070_D_AMAT_2 RTX3070_D_AMAT_25 RTX3070_D_AMAT_50 RTX3070_L1P RTX3070_L1_AMAT_2 RTX3070_L1_AMAT_25 RTX3070_L1_AMAT_50 RTX3070_L1_CacheSize_2 RTX3070_L1_CacheSize_4 RTX3070_L1_CacheSize_50 RTX3070_L2_AMAT_2 RTX3070_L2_AMAT_25 RTX3070_L2_AMAT_50 RTX3070_PCom RTX3070_PMem RTX3070_SM_23 RTX3070_SM_69 RTX3070_Init_2 RTX3070_Init_4 RTX3070_ALULat_25_gcstack RTX3070_ALULat_2_gcstack RTX3070_ALULat_50_gcstack RTX3070_AMAT_2 RTX3070_AMAT_25 RTX3070_AMAT_50]")
    parser.add_argument("-c",type=str, dest="cacheStatPath", default="sim_run_11.0",help="base path to accelsim_cache_stat_<config><suffix>.bin and kernel_cpi_<config><suffix>.bin. Sub-direcoty format should be <app>/<arg>/")
    parser.add_argument("-t",type=str, dest="tracePath"   , default="traces/rtx2060/11.0",help="base path to Accel-sim trace. Sub-direcoty format should be <app>/<arg>/traces/kernelslist.g")
    
    parser.add_argument("-r",type=str, dest="repWarpPath", default="",help="(Optional) rep warp import path. Import from [repWarpPath]/<app>/<arg>/rep_warp_RTX3070<suffix>.bin")
    parser.add_argument("-o",type=str, dest="outRepWarpPath", default="",help="(Optional) rep warp export path. Export to [outRepWarpPath]/<app>/<arg>/rep_warp_<config><suffix>.bin")

    parser.add_argument("-app",type=str   ,nargs='+', default=["all"],help="list of applications to execute [all | (Rodinia) b+tree backprop dwt gaus heart hotspot- hotspot3D hybrid lud nn- nw path stream bfs cfd kmeans srad_v1 srad_v2 (Parboil) sgemm stencil cutcp mri histo spmv (pannotia) color_max color_maxmin fw fw_block mis pagerank (Polybench) 2DConv 3DConv 3mm atax bicg -gemm gesummv mvt (NVIDIA SDK) Merse (Cactus) DCG GMS GRU LGT LMC LMR NST RFL SPT (Deepbench) conv_bench gemm_bench rnn_bench]")
    parser.add_argument("-n",type=int, dest="numCores"   , default=4,help="core number")
    parser.add_argument("-a",type=int, dest="schemeA", default=1,help="Apply SchemeA or not")

    parser.add_argument("-suffix",type=str, default="",help="suffix for the cache and cpi files")

    parser.add_argument("-no_run", action="store_true", default=False,help="do not run and just test script")
    parser.add_argument("-v", dest="verbose", action="store_true", default=False,help="verbose mode")
    scriptArgs = parser.parse_args()
    return scriptArgs

def ReadDefineAllApps():
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(FILE_DIR, "define-all-apps.yml")) as f:
        defineAllApps = yaml.load(f, Loader=yaml.FullLoader)
    appList = defineAllApps["APPARGS"]["execs"]
    for appDict in appList:
        for appName, argList in appDict.items():
            for arg in argList:
                argName = arg["args"]

                # switch all special characters of argName to _
                argName = re.sub(r"[^a-z^A-Z^0-9]", "_", str(argName).strip())
                # For every long arg lists - create a hash of the input args
                if len(str(argName)) > 256:
                    argName = "hashed_args_" + hashlib.md5(argName).hexdigest()

                if argName == "" or argName == 'None' or argName == None:
                        argName = "NO_ARGS"
                APP_ARG_DICT[appName] = argName

class RunWorkload:
    def __init__(self,command,binaryDir,tag):
        self.command = command
        self.binaryDir = binaryDir
        self.tag = tag
        self.startTime = 0
    
    def RunCommand(self):
        tmpDirReal = tempfile.TemporaryDirectory()
        tmpDir = tmpDirReal.name
        if scriptArgs.verbose:
            print(f"Tempdir {self.tag}: {tmpDir}")

        copyBinCommand = "cp -r %s/* %s"%(self.binaryDir,tmpDir)
        os.chdir(tmpDir)
        
        os.system(copyBinCommand)

        logfileName = "_".join(self.tag.split("/")) + ".log"
        self.command += " &> " + logfileName

        # Run command
        self.startTime = time.time()
        process = subprocess.Popen(self.command, shell=True, executable="/bin/bash")
        if scriptArgs.verbose:
            print(" %s (pid:%d)"%(self.command, process.pid))
        
        if scriptArgs.verbose:
            while True:
                time.sleep(3)
                returncode = process.poll()
                if returncode is not None:
                    break
                # occasionally print stdout&stderr
                os.system("tail -n 20 %s"%(logfileName))
        else:
            returncode = process.wait()

        with open(logfileName, "r") as f:
            outputMsg = f.readlines()

        if process.returncode != 0:
            return {self.tag:str(outputMsg)}

        resultDict = self.ParseStdout(outputMsg)

        logfilePath = os.path.join(FILE_DIR, "../logs", logfileName)
        with open(logfilePath, "w") as f:
            f.write("".join(outputMsg))

        tmpDirReal.cleanup()
        print(f"{self.tag} Done. Took {time.time()-self.startTime:.2f} seconds")
        return {self.tag:resultDict}
    
    def ParseStdout(self,lines):
        resultDict = dict()
        for i, line in enumerate(lines):
            if "Warning" in line or "Error" in line:
                print(f"{self.tag}: {line}")
                continue
            if not "Total Result" in line:
                continue
            keyList = line.split(',')
            valueList = lines[i+1].split(',')
        for key, value in zip(keyList[1:], valueList[1:]):
            resultDict[key.strip()] = float(value.strip())
        return resultDict

def RunCommandWrapper(runWorkload):
    return runWorkload.RunCommand()

def find_Latest_StatPath(app_path, prefix):
    
    file_list = glob.glob(os.path.join(app_path, f"{prefix}*"))
    
    sorted_file_list = sorted(file_list, key=lambda x : int(x[-17:-5].replace("_", "")))

    if sorted_file_list:
        return sorted_file_list[-1] 
    else:
        print(f"[Warning] Path does not exist... : {os.path.join(app_path, prefix)}")
        return None


if __name__ == "__main__":
    scriptArgs = get_args()

    if (READ_DEFINE_ALL_APPS):
        ReadDefineAllApps()
    
    RunWorkloads = [] # RunWorkload instance list
    if scriptArgs.configs == "all":
        scriptArgs.configs = CONFIGS
    for config in scriptArgs.configs:
        for appFullName, arg in APP_ARG_DICT.items():
            if not "all" in scriptArgs.app:
                # check if app has substring in scriptArgs.app list
                if not any(app in appFullName for app in scriptArgs.app):
                    continue

            configPath = os.path.abspath(os.path.join(FILE_DIR, "../configs", config+".config"))
            tracePath = os.path.abspath(os.path.join(scriptArgs.tracePath, appFullName, arg, "traces"))
            
            cacheStatPath = os.path.abspath(os.path.join(scriptArgs.cacheStatPath, appFullName, arg, "accelsim_cache_stat_" + config + scriptArgs.suffix + ".bin"))
            kernelCPIPath = os.path.abspath(os.path.join(scriptArgs.cacheStatPath, appFullName, arg, "kernel_cpi_" + config + scriptArgs.suffix + ".bin"))
            
            if scriptArgs.repWarpPath != "":
                repWarpPath = os.path.abspath(os.path.join(scriptArgs.repWarpPath, appFullName, arg, "rep_warp_" + "RTX3070" + scriptArgs.suffix + ".bin"))

            if cacheStatPath == None or not os.path.exists(cacheStatPath) or not os.path.exists(kernelCPIPath):
                print(f"[Warning] At path: {cacheStatPath}")
                print(f"[Warning] Cache statistic file, Kernel CPI file does not exist, skip {config} {appFullName}\n")
                continue
            elif scriptArgs.repWarpPath != "" and not os.path.exists(repWarpPath):
                print(f"[Warning] Rep warp file does not exist, skip {config} {appFullName}")
                continue

            if scriptArgs.outRepWarpPath != "":
                outRepWarpPath = os.path.abspath(os.path.join(scriptArgs.outRepWarpPath, appFullName, arg, "rep_warp_" + config + scriptArgs.suffix + ".bin"))
                dir = os.path.dirname(outRepWarpPath)
                # create directory hiearchy if not exist
                if not os.path.exists(dir):
                    os.makedirs(dir)

            command = "./GCoM"
            if scriptArgs.mode == 2:
                command += " -w 2"
            elif scriptArgs.mode == 3:
                command += " -w 3"
                command += " -r " + repWarpPath
            elif scriptArgs.mode == 4:
                command += " -w 4"
                command += " -k " + kernelCPIPath
            else:
                print("Invalid mode")
                exit(1)
                
            command += " -C " + configPath
            command += " -t " + tracePath
            command += " -c " + cacheStatPath

            if scriptArgs.outRepWarpPath != "":
                command += " -o " + outRepWarpPath
            
            if not scriptArgs.schemeA:
                command += " -a "
            
            RunWorkloads.append(RunWorkload(command, os.path.join(FILE_DIR, "../bin"), appFullName + "/" + arg + "/" + config))

    # Run workloads in parallel
    if not scriptArgs.no_run:
        logDir = os.path.join(FILE_DIR, "../logs")
        if not os.path.exists(logDir):
            os.makedirs(logDir)
        with Pool(scriptArgs.numCores) as pool:
            results = pool.map(RunCommandWrapper, RunWorkloads)
    else:
        for runWorkload in RunWorkloads:
            print(runWorkload.command)

    datas = []
    for result in results:
        # error check 
        for tag, resultDict in result.items():
            if type(resultDict) != dict:
                print("[Error] ", tag)
                print(resultDict)
                continue
            
            tag = tag.split("/")
            appArg = tag[0] + "/" + tag[1]
            config = tag[2]
            row = [appArg, "JHnoDram", config]
            columns = ["base", "comData", "comStruct", "memData", "memStruct", "controlHaz", "idle"]
            for columnName in columns:
                if not columnName in resultDict:
                    row.append(0)
                else:
                    row.append(resultDict[columnName])
            datas.append(row)
    
    if not os.path.exists(f'./csv'):
        os.makedirs(f'csv')

    df = pd.DataFrame(datas, columns=["app_arg", "method", "config"] + columns)
    df.to_csv('./csv/' + scriptArgs.suffix + scriptArgs.configs[0] + scriptArgs.app[0] + "_CPI.csv")
    
