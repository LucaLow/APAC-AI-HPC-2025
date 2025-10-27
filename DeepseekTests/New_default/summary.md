```Bash
[1,1]<stderr>:[2025-09-03 14:47:27 TP13] Config file not found at /scratch/users/industry/ai-hpc/apacsc03/py312/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json. Fallback to triton version 3.2.0 and use MoE kernel config from /scratch/users/industry/ai-hpc/apacsc03/py312/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=257,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json. Performance might be sub-optimal!
```
```


```Bash
[1,0]<stdout>:====== Offline Throughput Benchmark Result =======
[1,0]<stdout>:Backend:                                 engine
[1,0]<stdout>:Successful requests:                     2000
[1,0]<stdout>:Benchmark duration (s):                  161.80
[1,0]<stdout>:Total input tokens:                      626729
[1,0]<stdout>:Total generated tokens:                  388685
[1,0]<stdout>:Last generation throughput (tok/s):      39.32
[1,0]<stdout>:Request throughput (req/s):              12.36
[1,0]<stdout>:Input token throughput (tok/s):          3873.53
[1,0]<stdout>:Output token throughput (tok/s):         2402.29
[1,0]<stdout>:Total token throughput (tok/s):          6275.82
[1,0]<stdout>:==================================================
```