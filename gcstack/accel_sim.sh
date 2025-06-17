#!/bin/bash

# accel-sim git clone
git clone https://github.com/yeonan/accel-sim-framework.git
cd ./accel-sim-framework

# 환경 변수 설정
export CUDA_INSTALL_PATH=/usr/local/cuda-11.0
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

# gpu app git clone
git clone https://github.com/accel-sim/gpu-app-collection

# ./gpu-app-collection/get_data.sh 수정하면 빠른 실행 가능(이미 다운 받아놓았음)
# (전) wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/all.gpgpu-sim-app-data.tgz
# (후) cp /home/yeonan/all.gpgpu-sim-app-data.tgz $BASH_ROOT
source ./gpu-app-collection/src/setup_environment

# gpu app make
make -j -C ./gpu-app-collection/src rodinia_2.0-ft
make -C ./gpu-app-collection/src data

# SASS tracer make
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/

# 실제 device의 GPU에서 benchmark에 대한 SASS trace 수행
./util/tracer_nvbit/run_hw_trace.py -B rodinia_2.0-ft -D 0

# gpgpu-sim make
source ./gpu-simulator/setup_environment.sh
make -j -C ./gpu-simulator/

# ./util/job_launching/configs/define-standard-cfgs.yml 수정 필요
# (전) base_file: "$GPGPUSIM_ROOT/configs/tested-cfgs/Turing_RTX2060/gpgpusim.config"
# (후) base_file: "$GPGPUSIM_ROOT/configs/tested-cfgs/SM75_RTX2060/gpgpusim.config"


# 수행한 SASS trace를 RTX2060에서 추출했음을 가정하고 시뮬레이션
./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C RTX2060-SASS -T ./hw_run/traces/device-0/11.0/ -N myTest -l local

# status
./util/job_launching/job_status.py -N myTest -j procman

# 시뮬레이션 성공 여부 확인
./util/job_launching/monitor_func_test.py -v -N myTest

# 통계량 확인
./util/job_launching/get_stats.py -N myTest (-K -k) | tee stats.csv 


# nvprof disable and nsight profiler 사용 (sudo 권한 필요)
sudo -s
export CUDA_INSTALL_PATH=/usr/local/cuda-11.0
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export GPUAPPS_ROOT="/home/jounghoo/titanrtx_JH/accel-sim-framework/gpu-app-collection"
./util/hw_stats/run_hw.py -B rodinia_2.0-ft --nsight_profiler --disable_nvprof -C other_stats
exit

# 통계량 추출
./util/job_launching/get_stats.py -R -k -K -N myTest | tee per.kernel.stats.csv

# correlation 확인
./util/plotting/plot-correlation.py -c per.kernel.stats.csv -H ./hw_run/device-0/11.2/
