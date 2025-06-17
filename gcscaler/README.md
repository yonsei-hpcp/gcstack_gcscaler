# GCScaler

GCScaler leverages the existing GPU interval model, [GCoM \[ISCA '22\]](https://doi.org/10.1145/3470496.3527384), and calibrates representative warps using the baseline CPI information obtained from GCStack to derive scaling values for constructing the CPI stacks of alternative GPU designs.

## Dependencies
gcc/g++ 8.4.0 or higher
```
sudo apt install xutils-dev bison zlib1g-dev flex libglu1-mesa-dev libssl-dev libxml2-dev libxml2-dev git

# scripts under utils/
pip install pandas PyYAML
```
## zlib & gzstream
```bash
cd third-party/zlib-1.2.12
prefix=. ./configure
make test
make install prefix=.

cd ../gzstream
make test
make
```

## boost serialization
```
cd third-party/boost_1_86_0/
./bootstrap.sh
./b2 --with-serialization
```

# Build & Run

```
$ make -j [debug]
$ ./bin/GCoM -h
usage : ./bin/GCoM 
 -h    help
 -w [INT] work ID
 -C [path] configuration file path
 -t [path] benchmark trace directory
 -a (Optional) Use GCoM idle definition which only accounts active SMs

[work 0] RunGCoMWithAccelsimCacheModel options
 -c [path] cache statistics trace path
 -r [path] representative warp index trace path

[work 2] RunGCoMWithAccelsimCacheModelGPUMechRepWarpSelect options
 -c [path] cache statistics trace path

[work 3] RunGCoMWithAccelsimCacheModelCPIAssistedKmeansRepWarp options
 -c [path] cache statistics trace path
 -r [path] representative warp index trace path

[work 4] RunGCoMWithAccelsimCacheModelGCstackBaseRepWarpSearch options
 -c [path] cache statistics trace path
 -k [path] kernel target CPI Stack file path for representative warp search
 -o [path] (Optional) export selected representative warp index to [path]

# Example
# Export calibrated representative warp with GCStack's CPI information
$ ./bin/GCoM -w 4 -C configs/RTX3070.config -t <trace_path> -c <cache_statistics_path_from_GCStack> -k <kernel_cpi_path_from_GCStack>

# Run with calibrated representative warp
$ ./bin/GCoM -w 3 -C configs/RTX3070.config -t <trace_path> -c <cache_statistics_path_from_GCStack> -k <kernel_cpi_path_from_GCStack>
...
K1 Result       ,numThreadInst  ,numCycle       ,cpi    ,base   ,comData        ,comStruct      ,memData        ,memStruct      ,idle
Value   ,1327636        ,2321   ,0.0017487      ,0.000269954    ,8.44526e-05    ,0.00018843     ,0.00112356     ,3.03547e-05    ,5.19526e-05
Total Result    ,numThreadInst  ,numCycle       ,cpi    ,base   ,comData        ,comStruct      ,memData        ,memStruct      ,idle
Value   ,1327636        ,2321   ,0.00174822     ,0.000269954    ,8.44526e-05    ,0.00018843     ,0.00112356     ,3.03547e-05    ,5.19526e-05

```

# File structures
```
configs/ : hardware configuration file
src/ : source code of performance model and cache simulator. Modified for calibration.
third-party/ : code from other projects. Modified based on our needs.
```

