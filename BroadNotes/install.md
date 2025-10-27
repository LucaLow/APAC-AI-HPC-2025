Forgot to document the exact installation, it was something like:
```Bash
uv venv --python 3.12
source ~/scratch/py312/bin/activate
uv pip install 'sglang[all]' # I think I did srt but then did all after.

# Had an issue with these two (may not be required)
uv pip install ninja cmake

# Download the cuda 12.9 api for nvcc (mostly deepgemm optimisations) 
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
```

on GPU node (Login node will cauase segfault):
```Bash
sh cuda_12.9.0_575.51.03_linux.run --toolkit --silent --override --installpath=$HOME/cuda-12.9
```

To use the updated cuda api and shit:
```Bash
export CUDA_HOME=$HOME/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```