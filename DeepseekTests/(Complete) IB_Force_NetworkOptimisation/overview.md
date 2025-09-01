# Job Description
This started as to an investigation in the error logs of the default runs, there appeared to be network issues where there were network interface clashes or some shit, so I figured that forcing Infiniband would help

# Findings:
It was found that restricting so that the job only used one channel (the ethernet channel) slowed down the program alot.
It was found that restricting so that the job only used one channel (the infiniband channel) slowed down the program a little bit.

I discovered I was restricting to only one channel, broadened the scope so that I was using all the infiniband channels.
This significantly improved performance, so that the program consistantly runs ~x faster.
Iteration3 is different from iteration4 & 5 though may be functionally the same, further testing would be required though is unlikely to be beneficial.

Baseline: (Auto Inifinband/Ethernet)
```Bash
====== Offline Throughput Benchmark Result =======
Backend:                                 engine    
Successful requests:                     100       
Benchmark duration (s):                  45.40     
Total input tokens:                      34308     
Total generated tokens:                  21395     
Last generation throughput (tok/s):      43.26     
Request throughput (req/s):              2.20      
Input token throughput (tok/s):          755.68    
Output token throughput (tok/s):         471.25    
Total token throughput (tok/s):          1226.93   
==================================================
```

Run 1: (Infininband Complete IT4)
```Bash
====== Offline Throughput Benchmark Result =======
Backend:                                 engine    
Successful requests:                     100       
Benchmark duration (s):                  44.26     
Total input tokens:                      34308     
Total generated tokens:                  21395     
Last generation throughput (tok/s):      43.96     
Request throughput (req/s):              2.26      
Input token throughput (tok/s):          775.18    
Output token throughput (tok/s):         483.41    
Total token throughput (tok/s):          1258.59   
==================================================
```

Run 2: (Infiniband Complete IT5)
```Bash
====== Offline Throughput Benchmark Result =======
Backend:                                 engine    
Successful requests:                     100       
Benchmark duration (s):                  44.71     
Total input tokens:                      34308     
Total generated tokens:                  21395     
Last generation throughput (tok/s):      43.07     
Request throughput (req/s):              2.24      
Input token throughput (tok/s):          767.40    
Output token throughput (tok/s):         478.56    
Total token throughput (tok/s):          1245.96   
==================================================
```