# One GPU is enough
git clone https://github.com/sgl-project/sglang.git
cd sglang/benchmark/kernels/fused_moe_triton
python3 tuning_fused_moe_triton.py \
  --model /path/to/deepseek-r1 --dtype fp8_w8a8 --tp 16 --tune

# Put the produced JSON under the exact dir your warning printed, e.g.:
CONF_DIR=/scratch/.../site-packages/sglang/.../configs/triton_3_3_1
mkdir -p "$CONF_DIR"
cp "E=257,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json" "$CONF_DIR/"


scratch/py312/bin/python sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py   --model /path/to/deepseek-r1 --dtype bfloat16 --tp 16 --tune