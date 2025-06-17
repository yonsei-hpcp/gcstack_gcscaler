# GCStack

GCStack is implemented on top of [Accel-Sim](https://github.com/accel-sim/accel-sim-framework). Its basic dependencies and execution method are the same as those of Accel-Sim.

## Dependencies

```bash
apt-get install -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++-7 g++-7 libssl-dev libxml2-dev \
      libxml2-dev vim python-setuptools python3-pip

pip install pyyaml psutil
```

- Ubuntu 20.04 or higher

- gcc/g++ 7.5.0

- CUDA 11.0

- Boost C++ Libraries - 1.65.0

## Build & Run

### Collect SASS trace

Please refer to [Accel-sim Tracer](https://github.com/accel-sim/accel-sim-framework/tree/2260456ea5e6a1420f5734f145a4b7d8ab1d4737) for SASS trace collection. Then, compress resulting `*.traceg` files to `*traceg.gz` using `gzip`.

### Build

```bash
export CUDA_INSTALL_PATH=<your_cuda_path> 
export PATH=$CUDA_INSTALL_PATH/bin:$PATH 
export LD_LIBRARY_PATH=<your_libboost_path>:$LD_LIBRARY_PATH

source ./gpu-simulator/setup_environment.sh

make -j -C ./gpu-simulator/
```

This will produce executable in:
```bash
./gpu-simulator/bin/release/accel-sim.out
```

### Run
To generate cache stats and kernel CPI information for GCScaler's interval model (GCoM), please run `./util/job_launching/run_simulations.py`.

```bash
python3 ./util/job_launching/run_simulations.py -B <benchmark> -C <config> -T <SASS trace path> -N <launch name> -l <launcher>
```

After execution, all output files will be generated in the `./sim_run_11.0`. This directory contains simulation, cache stats (`accelsim_cache_stat_<config><launch name>.bin`), and kernel cpi information (`kernel_cpi_<config><launch name>.bin`).

The GCStack CPI stack will be printed in simulation output files like below:
```bash
========= GCStack CPI Stack Information =========

GCStack_Base:22617.043478
GCStack_MemStruct:2079.921239
GCStack_MemData:9919.065217
GCStack_Sync:2435.597826
GCStack_ComStruct:12827.122239
GCStack_ComData:3805.184783
GCStack_Control:0.000000
GCStack_Idle:1958.065217
GCStack_Total:55642.000000


========= Idle sub-classification =========

GCStack_EmptyWS:302.697011
GCStack_IdleSubcore:6.076087
GCStack_IdleSM:1648.847826
GCStack_IdleSubClassificationTotal:1957.620924
```
