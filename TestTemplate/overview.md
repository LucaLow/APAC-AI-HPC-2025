# Job Description

# Benchmark Results:
```Bash

```

# Findings:

qsub -I -l select=1:ncpus=16:ngpus=1:mem=80gb -l walltime=00:15:00 -P 50000097


Pre:
```Bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```

Post:
```Bash
apacsc03@a2ap-dgx037:~$ "${HOME}/scratch/py312/bin/nvcc" --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```



Build SGlang:
```Bash
apacsc03@a2ap-dgx038:~/scratch$ # Make sure your env sees nvcc 12.9 first
export CUDA_HOME=$HOME/scratch/cuda129       # or wherever you installed it
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas
export NVIDIA_NVJITLINK_PATH=$CUDA_HOME/lib64

# Force reinstall sglang from PyPI, rebuilding C++/CUDA extensions
${HOME}/scratch/py312/bin/pip install --no-binary=sglang --force-reinstall "sglang[all]"
```