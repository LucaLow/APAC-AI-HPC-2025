[1,0]<stderr>:Traceback (most recent call last):
[1,0]<stderr>:  File "<frozen runpy>", line 198, in _run_module_as_main
[1,0]<stderr>:  File "<frozen runpy>", line 88, in _run_code
[1,0]<stderr>:  File "/home/users/industry/ai-hpc/apacsc03/scratch/py312/lib/python3.12/site-packages/sglang/bench_offline_throughput.py", line 441, in <module>
[1,0]<stderr>:    server_args = ServerArgs.from_cli_args(args)
[1,0]<stderr>:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[1,0]<stderr>:  File "/home/users/industry/ai-hpc/apacsc03/scratch/py312/lib/python3.12/site-packages/sglang/srt/server_args.py", line 1890, in from_cli_args
[1,0]<stderr>:    return cls(**{attr: getattr(args, attr) for attr in attrs})
[1,0]<stderr>:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[1,0]<stderr>:  File "<string>", line 188, in __init__
[1,0]<stderr>:  File "/home/users/industry/ai-hpc/apacsc03/scratch/py312/lib/python3.12/site-packages/sglang/srt/server_args.py", line 460, in __post_init__
[1,0]<stderr>:    self.dp_size > 1
[1,0]<stderr>:AssertionError: Please set a dp-size > 1. You can use 1 < dp-size <= tp-size
[1,1]<stderr>:Traceback (most recent call last):
[1,1]<stderr>:  File "<frozen runpy>", line 198, in _run_module_as_main
[1,1]<stderr>:  File "<frozen runpy>", line 88, in _run_code
[1,1]<stderr>:  File "/home/users/industry/ai-hpc/apacsc03/scratch/py312/lib/python3.12/site-packages/sglang/bench_offline_throughput.py", line 441, in <module>
[1,1]<stderr>:    server_args = ServerArgs.from_cli_args(args)
[1,1]<stderr>:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[1,1]<stderr>:  File "/home/users/industry/ai-hpc/apacsc03/scratch/py312/lib/python3.12/site-packages/sglang/srt/server_args.py", line 1890, in from_cli_args
[1,1]<stderr>:    return cls(**{attr: getattr(args, attr) for attr in attrs})
[1,1]<stderr>:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[1,1]<stderr>:  File "<string>", line 188, in __init__
[1,1]<stderr>:  File "/home/users/industry/ai-hpc/apacsc03/scratch/py312/lib/python3.12/site-packages/sglang/srt/server_args.py", line 460, in __post_init__
[1,1]<stderr>:    self.dp_size > 1
[1,1]<stderr>:AssertionError: Please set a dp-size > 1. You can use 1 < dp-size <= tp-size