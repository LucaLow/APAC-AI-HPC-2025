# Job Description
--DP 2 
--TP 8
--num-prompts 100

(Updated):
```Shell
+ --enable-dp-attention
+ --enable-dp-lm-head
```

Original: 81929.pbs111
Updated: 81949.pbs111 
# Benchmark Results:
```Bash

```

# Findings:
**Data Parallelism does not appear possible due to the model size being too large for 16 H100's**
