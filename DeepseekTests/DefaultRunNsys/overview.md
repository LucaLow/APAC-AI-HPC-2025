# Job Description
This was just a regular test to determine a baseline for running deepseek across 16 Gpu's on 2 nodes. This test used Nvidia's Nsys, an additional test is queued without nsys for a more accurate baseline.

# Benchmark Results:
```Bash
[1,0]<stdout>:#Input tokens: 1972
[1,0]<stdout>:#Output tokens: 2784
[1,0]<stdout>:#Input tokens: 2560
[1,0]<stdout>:#Output tokens: 160
[1,0]<stdout>:
[1,0]<stdout>:====== Offline Throughput Benchmark Result =======
[1,0]<stdout>:Backend:                                 engine    
[1,0]<stdout>:Successful requests:                     10        
[1,0]<stdout>:Benchmark duration (s):                  19.94     
[1,0]<stdout>:Total input tokens:                      1972      
[1,0]<stdout>:Total generated tokens:                  2784      
[1,0]<stdout>:Last generation throughput (tok/s):      32.09     
[1,0]<stdout>:Request throughput (req/s):              0.50      
[1,0]<stdout>:Input token throughput (tok/s):          98.89     
[1,0]<stdout>:Output token throughput (tok/s):         139.61    
[1,0]<stdout>:Total token throughput (tok/s):          238.50    
[1,0]<stdout>:==================================================
```

# Findings:
## Errors at End of Logs
There appears to be a whole load of errors at the end of the program, further investigation reveals these are most likely related to node 0 quitting smoothly and sending EOL messages to node 1, node 1 doesn't take that very well, hangs or something, node 0 doesn't give a fuck and finishes, and then the time limit is reached as node 1 continues to hang, resulting in the running over time errors.

## KV Cache Pool Findings
[Sglang Suggestions Prompting Investigation](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html#tune-mem-fraction-static-to-increase-kv-cache-pool-capacity)
Sglang reccomends the following to maximise concurency:
Check the available_gpu_mem value.
- If it is between 5–8 GB, the setting is good.
- If it is too high (e.g., 10 - 20 GB), increase --mem-fraction-static to allocate more memory to the KV cache.
- If it is too low, you risk out-of-memory (OOM) errors later, so decrease --mem-fraction-static.
Another straightforward approach is to increase --mem-fraction-static in increments of 0.01 until you encounter OOM errors for your workloads.

**The following was found in the output logs:**
> max_total_num_tokens=294999, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=2048, context_len=163840, available_gpu_mem=14.11 GB

This indicates that we should increase the --mem-fraction-static but I'll make further larger changes first.