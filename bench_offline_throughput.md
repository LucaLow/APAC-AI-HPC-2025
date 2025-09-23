All bench_offline_throughput parameters:

usage: bench_offline_throughput.py [-h] --model-path MODEL_PATH [--tokenizer-path TOKENIZER_PATH]
                                   [--tokenizer-mode {auto,slow}]
                                   [--tokenizer-worker-num TOKENIZER_WORKER_NUM] [--skip-tokenizer-init]
                                   [--load-format {auto,pt,safetensors,npcache,dummy,sharded_state,gguf,bitsandbytes,layered,remote}]
                                   [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
                                   [--trust-remote-code] [--context-length CONTEXT_LENGTH]
                                   [--is-embedding] [--enable-multimodal] [--revision REVISION]
                                   [--model-impl MODEL_IMPL] [--host HOST] [--port PORT]
                                   [--skip-server-warmup] [--warmups WARMUPS] [--nccl-port NCCL_PORT]
                                   [--dtype {auto,half,float16,bfloat16,float,float32}]
                                   [--quantization {awq,fp8,gptq,marlin,gptq_marlin,awq_marlin,bitsandbytes,gguf,modelopt,modelopt_fp4,petit_nvfp4,w8a8_int8,w8a8_fp8,moe_wna16,qoq,w4afp8,mxfp4}]
                                   [--quantization-param-path QUANTIZATION_PARAM_PATH]
                                   [--kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3}]
                                   [--mem-fraction-static MEM_FRACTION_STATIC]
                                   [--max-running-requests MAX_RUNNING_REQUESTS]
                                   [--max-queued-requests MAX_QUEUED_REQUESTS]
                                   [--max-total-tokens MAX_TOTAL_TOKENS]
                                   [--chunked-prefill-size CHUNKED_PREFILL_SIZE]
                                   [--max-prefill-tokens MAX_PREFILL_TOKENS]
                                   [--schedule-policy {lpm,random,fcfs,dfs-weight,lof}]
                                   [--schedule-conservativeness SCHEDULE_CONSERVATIVENESS]
                                   [--page-size PAGE_SIZE]
                                   [--hybrid-kvcache-ratio [HYBRID_KVCACHE_RATIO]]
                                   [--swa-full-tokens-ratio SWA_FULL_TOKENS_RATIO]
                                   [--disable-hybrid-swa-memory] [--device DEVICE]
                                   [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
                                   [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
                                   [--max-micro-batch-size MAX_MICRO_BATCH_SIZE]
                                   [--stream-interval STREAM_INTERVAL] [--stream-output]
                                   [--random-seed RANDOM_SEED]
                                   [--constrained-json-whitespace-pattern CONSTRAINED_JSON_WHITESPACE_PATTERN]
                                   [--watchdog-timeout WATCHDOG_TIMEOUT] [--dist-timeout DIST_TIMEOUT]
                                   [--download-dir DOWNLOAD_DIR] [--base-gpu-id BASE_GPU_ID]
                                   [--gpu-id-step GPU_ID_STEP] [--sleep-on-idle] [--log-level LOG_LEVEL]
                                   [--log-level-http LOG_LEVEL_HTTP] [--log-requests]
                                   [--log-requests-level {0,1,2,3}]
                                   [--crash-dump-folder CRASH_DUMP_FOLDER] [--show-time-cost]
                                   [--enable-metrics] [--enable-metrics-for-all-schedulers]
                                   [--bucket-time-to-first-token BUCKET_TIME_TO_FIRST_TOKEN [BUCKET_TIME_TO_FIRST_TOKEN ...]]
                                   [--bucket-inter-token-latency BUCKET_INTER_TOKEN_LATENCY [BUCKET_INTER_TOKEN_LATENCY ...]]
                                   [--bucket-e2e-request-latency BUCKET_E2E_REQUEST_LATENCY [BUCKET_E2E_REQUEST_LATENCY ...]]
                                   [--collect-tokens-histogram]
                                   [--prompt-tokens-buckets PROMPT_TOKENS_BUCKETS [PROMPT_TOKENS_BUCKETS ...]]
                                   [--generation-tokens-buckets GENERATION_TOKENS_BUCKETS [GENERATION_TOKENS_BUCKETS ...]]
                                   [--gc-warning-threshold-secs GC_WARNING_THRESHOLD_SECS]
                                   [--decode-log-interval DECODE_LOG_INTERVAL]
                                   [--enable-request-time-stats-logging]
                                   [--kv-events-config KV_EVENTS_CONFIG] [--api-key API_KEY]
                                   [--served-model-name SERVED_MODEL_NAME]
                                   [--weight-version WEIGHT_VERSION] [--chat-template CHAT_TEMPLATE]
                                   [--completion-template COMPLETION_TEMPLATE]
                                   [--file-storage-path FILE_STORAGE_PATH] [--enable-cache-report]
                                   [--reasoning-parser {deepseek-r1,deepseek-v3,glm45,gpt-oss,kimi,qwen3,qwen3-thinking,step3}]
                                   [--tool-call-parser {llama3,qwen25,mistral,deepseekv3,deepseekv31,pythonic,kimi_k2,qwen3_coder,glm45,step3,gpt-oss}]
                                   [--tool-server TOOL_SERVER] [--data-parallel-size DATA_PARALLEL_SIZE]
                                   [--load-balance-method {round_robin,shortest_queue,minimum_tokens}]
                                   [--prefill-round-robin-balance] [--dist-init-addr DIST_INIT_ADDR]
                                   [--nnodes NNODES] [--node-rank NODE_RANK]
                                   [--json-model-override-args JSON_MODEL_OVERRIDE_ARGS]
                                   [--preferred-sampling-params PREFERRED_SAMPLING_PARAMS]
                                   [--enable-lora] [--max-lora-rank MAX_LORA_RANK]
                                   [--lora-target-modules [{q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv_proj,gate_up_proj,all} ...]]
                                   [--lora-paths [LORA_PATHS ...]]
                                   [--max-loras-per-batch MAX_LORAS_PER_BATCH]
                                   [--max-loaded-loras MAX_LOADED_LORAS] [--lora-backend LORA_BACKEND]
                                   [--attention-backend {triton,torch_native,cutlass_mla,fa3,flashinfer,flashmla,trtllm_mla,trtllm_mha,dual_chunk_flash_attn,hybrid_linear_attn,aiter,wave,intel_amx,ascend}]
                                   [--prefill-attention-backend {triton,torch_native,cutlass_mla,fa3,flashinfer,flashmla,trtllm_mla,trtllm_mha,dual_chunk_flash_attn,hybrid_linear_attn,aiter,wave,intel_amx,ascend}]
                                   [--decode-attention-backend {triton,torch_native,cutlass_mla,fa3,flashinfer,flashmla,trtllm_mla,trtllm_mha,dual_chunk_flash_attn,hybrid_linear_attn,aiter,wave,intel_amx,ascend}]
                                   [--sampling-backend {flashinfer,pytorch}]
                                   [--grammar-backend {xgrammar,outlines,llguidance,none}]
                                   [--mm-attention-backend {sdpa,fa3,triton_attn}]
                                   [--speculative-algorithm {EAGLE,EAGLE3,NEXTN,STANDALONE}]
                                   [--speculative-draft-model-path SPECULATIVE_DRAFT_MODEL_PATH]
                                   [--speculative-draft-model-revision SPECULATIVE_DRAFT_MODEL_REVISION]
                                   [--speculative-num-steps SPECULATIVE_NUM_STEPS]
                                   [--speculative-eagle-topk SPECULATIVE_EAGLE_TOPK]
                                   [--speculative-num-draft-tokens SPECULATIVE_NUM_DRAFT_TOKENS]
                                   [--speculative-accept-threshold-single SPECULATIVE_ACCEPT_THRESHOLD_SINGLE]
                                   [--speculative-accept-threshold-acc SPECULATIVE_ACCEPT_THRESHOLD_ACC]
                                   [--speculative-token-map SPECULATIVE_TOKEN_MAP]
                                   [--speculative-attention-mode {prefill,decode}]
                                   [--expert-parallel-size EXPERT_PARALLEL_SIZE]
                                   [--moe-a2a-backend {none,deepep}]
                                   [--moe-runner-backend {auto,triton,triton_kernel,flashinfer_trtllm,flashinfer_cutlass,flashinfer_mxfp4}]
                                   [--flashinfer-mxfp4-moe-precision {default,bf16}]
                                   [--enable-flashinfer-allreduce-fusion]
                                   [--deepep-mode {normal,low_latency,auto}]
                                   [--ep-num-redundant-experts EP_NUM_REDUNDANT_EXPERTS]
                                   [--ep-dispatch-algorithm EP_DISPATCH_ALGORITHM]
                                   [--init-expert-location INIT_EXPERT_LOCATION] [--enable-eplb]
                                   [--eplb-algorithm EPLB_ALGORITHM]
                                   [--eplb-rebalance-num-iterations EPLB_REBALANCE_NUM_ITERATIONS]
                                   [--eplb-rebalance-layers-per-chunk EPLB_REBALANCE_LAYERS_PER_CHUNK]
                                   [--eplb-min-rebalancing-utilization-threshold EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD]
                                   [--expert-distribution-recorder-mode EXPERT_DISTRIBUTION_RECORDER_MODE]
                                   [--expert-distribution-recorder-buffer-size EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE]
                                   [--enable-expert-distribution-metrics]
                                   [--deepep-config DEEPEP_CONFIG]
                                   [--moe-dense-tp-size MOE_DENSE_TP_SIZE]
                                   [--max-mamba-cache-size MAX_MAMBA_CACHE_SIZE]
                                   [--mamba-ssm-dtype {float32,bfloat16}] [--enable-hierarchical-cache]
                                   [--hicache-ratio HICACHE_RATIO] [--hicache-size HICACHE_SIZE]
                                   [--hicache-write-policy {write_back,write_through,write_through_selective}]
                                   [--hicache-io-backend {direct,kernel}]
                                   [--hicache-mem-layout {layer_first,page_first}]
                                   [--hicache-storage-backend {file,mooncake,hf3fs,nixl}]
                                   [--hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}]
                                   [--hicache-storage-backend-extra-config HICACHE_STORAGE_BACKEND_EXTRA_CONFIG]
                                   [--enable-lmcache] [--enable-double-sparsity]
                                   [--ds-channel-config-path DS_CHANNEL_CONFIG_PATH]
                                   [--ds-heavy-channel-num DS_HEAVY_CHANNEL_NUM]
                                   [--ds-heavy-token-num DS_HEAVY_TOKEN_NUM]
                                   [--ds-heavy-channel-type DS_HEAVY_CHANNEL_TYPE]
                                   [--ds-sparse-decode-threshold DS_SPARSE_DECODE_THRESHOLD]
                                   [--cpu-offload-gb CPU_OFFLOAD_GB]
                                   [--offload-group-size OFFLOAD_GROUP_SIZE]
                                   [--offload-num-in-group OFFLOAD_NUM_IN_GROUP]
                                   [--offload-prefetch-step OFFLOAD_PREFETCH_STEP]
                                   [--offload-mode OFFLOAD_MODE] [--disable-radix-cache]
                                   [--cuda-graph-max-bs CUDA_GRAPH_MAX_BS]
                                   [--cuda-graph-bs CUDA_GRAPH_BS [CUDA_GRAPH_BS ...]]
                                   [--disable-cuda-graph] [--disable-cuda-graph-padding]
                                   [--enable-profile-cuda-graph] [--enable-cudagraph-gc]
                                   [--enable-nccl-nvls] [--enable-symm-mem]
                                   [--disable-flashinfer-cutlass-moe-fp4-allgather]
                                   [--enable-tokenizer-batch-encode] [--disable-outlines-disk-cache]
                                   [--disable-custom-all-reduce] [--enable-mscclpp]
                                   [--disable-overlap-schedule] [--enable-mixed-chunk]
                                   [--enable-dp-attention] [--enable-dp-lm-head]
                                   [--enable-two-batch-overlap]
                                   [--tbo-token-distribution-threshold TBO_TOKEN_DISTRIBUTION_THRESHOLD]
                                   [--enable-torch-compile]
                                   [--torch-compile-max-bs TORCH_COMPILE_MAX_BS]
                                   [--torchao-config TORCHAO_CONFIG] [--enable-nan-detection]
                                   [--enable-p2p-check] [--triton-attention-reduce-in-fp32]
                                   [--triton-attention-num-kv-splits TRITON_ATTENTION_NUM_KV_SPLITS]
                                   [--num-continuous-decode-steps NUM_CONTINUOUS_DECODE_STEPS]
                                   [--delete-ckpt-after-loading] [--enable-memory-saver]
                                   [--allow-auto-truncate] [--enable-custom-logit-processor]
                                   [--flashinfer-mla-disable-ragged] [--disable-shared-experts-fusion]
                                   [--disable-chunked-prefix-cache] [--disable-fast-image-processor]
                                   [--enable-return-hidden-states]
                                   [--scheduler-recv-interval SCHEDULER_RECV_INTERVAL]
                                   [--numa-node NUMA_NODE [NUMA_NODE ...]]
                                   [--debug-tensor-dump-output-folder DEBUG_TENSOR_DUMP_OUTPUT_FOLDER]
                                   [--debug-tensor-dump-input-file DEBUG_TENSOR_DUMP_INPUT_FILE]
                                   [--debug-tensor-dump-inject DEBUG_TENSOR_DUMP_INJECT]
                                   [--debug-tensor-dump-prefill-only]
                                   [--disaggregation-mode {null,prefill,decode}]
                                   [--disaggregation-transfer-backend {mooncake,nixl,ascend,fake}]
                                   [--disaggregation-bootstrap-port DISAGGREGATION_BOOTSTRAP_PORT]
                                   [--disaggregation-decode-tp DISAGGREGATION_DECODE_TP]
                                   [--disaggregation-decode-dp DISAGGREGATION_DECODE_DP]
                                   [--disaggregation-prefill-pp DISAGGREGATION_PREFILL_PP]
                                   [--disaggregation-ib-device DISAGGREGATION_IB_DEVICE]
                                   [--num-reserved-decode-tokens NUM_RESERVED_DECODE_TOKENS]
                                   [--custom-weight-loader [CUSTOM_WEIGHT_LOADER ...]]
                                   [--weight-loader-disable-mmap] [--enable-pdmux]
                                   [--sm-group-num SM_GROUP_NUM] [--enable-ep-moe] [--enable-deepep-moe]
                                   [--enable-flashinfer-cutlass-moe] [--enable-flashinfer-trtllm-moe]
                                   [--enable-triton-kernel-moe] [--enable-flashinfer-mxfp4-moe]
                                   [--backend BACKEND] [--result-filename RESULT_FILENAME]
                                   [--dataset-name {sharegpt,random,generated-shared-prefix}]
                                   [--dataset-path DATASET_PATH] [--num-prompts NUM_PROMPTS]
                                   [--sharegpt-output-len SHAREGPT_OUTPUT_LEN]
                                   [--sharegpt-context-len SHAREGPT_CONTEXT_LEN]
                                   [--random-input-len RANDOM_INPUT_LEN]
                                   [--random-output-len RANDOM_OUTPUT_LEN]
                                   [--random-range-ratio RANDOM_RANGE_RATIO]
                                   [--gsp-num-groups GSP_NUM_GROUPS]
                                   [--gsp-prompts-per-group GSP_PROMPTS_PER_GROUP]
                                   [--gsp-system-prompt-len GSP_SYSTEM_PROMPT_LEN]
                                   [--gsp-question-len GSP_QUESTION_LEN]
                                   [--gsp-output-len GSP_OUTPUT_LEN] [--seed SEED]
                                   [--disable-ignore-eos]
                                   [--extra-request-body {"key1": "value1", "key2": "value2"}]
                                   [--apply-chat-template] [--profile] [--skip-warmup] [--do-not-exit]
                                   [--prompt-suffix PROMPT_SUFFIX]

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH, --model MODEL_PATH
                        The path of the model weights. This can be a local folder or a Hugging Face repo
                        ID.
  --tokenizer-path TOKENIZER_PATH
                        The path of the tokenizer.
  --tokenizer-mode {auto,slow}
                        Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will
                        always use the slow tokenizer.
  --tokenizer-worker-num TOKENIZER_WORKER_NUM
                        The worker num of the tokenizer manager.
  --skip-tokenizer-init
                        If set, skip init tokenizer and pass input_ids in generate request.
  --load-format {auto,pt,safetensors,npcache,dummy,sharded_state,gguf,bitsandbytes,layered,remote}
                        The format of the model weights to load. "auto" will try to load the weights in
                        the safetensors format and fall back to the pytorch bin format if safetensors
                        format is not available. "pt" will load the weights in the pytorch bin format.
                        "safetensors" will load the weights in the safetensors format. "npcache" will
                        load the weights in pytorch format and store a numpy cache to speed up the
                        loading. "dummy" will initialize the weights with random values, which is mainly
                        for profiling."gguf" will load the weights in the gguf format. "bitsandbytes"
                        will load the weights using bitsandbytes quantization."layered" loads weights
                        layer by layer so that one can quantize a layer before loading another to make
                        the peak memory envelope smaller.
  --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
                        Extra config for model loader. This will be passed to the model loader
                        corresponding to the chosen load_format.
  --trust-remote-code   Whether or not to allow for custom models defined on the Hub in their own
                        modeling files.
  --context-length CONTEXT_LENGTH
                        The model's maximum context length. Defaults to None (will use the value from
                        the model's config.json instead).
  --is-embedding        Whether to use a CausalLM as an embedding model.
  --enable-multimodal   Enable the multimodal functionality for the served model. If the model being
                        served is not multimodal, nothing will happen
  --revision REVISION   The specific model version to use. It can be a branch name, a tag name, or a
                        commit id. If unspecified, will use the default version.
  --model-impl MODEL_IMPL
                        Which implementation of the model to use. * "auto" will try to use the SGLang
                        implementation if it exists and fall back to the Transformers implementation if
                        no SGLang implementation is available. * "sglang" will use the SGLang model
                        implementation. * "transformers" will use the Transformers model implementation.
  --host HOST           The host of the HTTP server.
  --port PORT           The port of the HTTP server.
  --skip-server-warmup  If set, skip warmup.
  --warmups WARMUPS     Specify custom warmup functions (csv) to run before server starts eg.
                        --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and
                        `warmup_name2` specified in warmup.py before the server starts listening for
                        requests
  --nccl-port NCCL_PORT
                        The port for NCCL distributed environment setup. Defaults to a random port.
  --dtype {auto,half,float16,bfloat16,float,float32}
                        Data type for model weights and activations. * "auto" will use FP16 precision
                        for FP32 and FP16 models, and BF16 precision for BF16 models. * "half" for FP16.
                        Recommended for AWQ quantization. * "float16" is the same as "half". *
                        "bfloat16" for a balance between precision and range. * "float" is shorthand for
                        FP32 precision. * "float32" for FP32 precision.
  --quantization {awq,fp8,gptq,marlin,gptq_marlin,awq_marlin,bitsandbytes,gguf,modelopt,modelopt_fp4,petit_nvfp4,w8a8_int8,w8a8_fp8,moe_wna16,qoq,w4afp8,mxfp4}
                        The quantization method.
  --quantization-param-path QUANTIZATION_PARAM_PATH
                        Path to the JSON file containing the KV cache scaling factors. This should
                        generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling
                        factors default to 1.0, which may cause accuracy issues.
  --kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3}
                        Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" and
                        "fp8_e4m3" is supported for CUDA 11.8+.
  --mem-fraction-static MEM_FRACTION_STATIC
                        The fraction of the memory used for static allocation (model weights and KV
                        cache memory pool). Use a smaller value if you see out-of-memory errors.
  --max-running-requests MAX_RUNNING_REQUESTS
                        The maximum number of running requests.
  --max-queued-requests MAX_QUEUED_REQUESTS
                        The maximum number of queued requests. This option is ignored when using
                        disaggregation-mode.
  --max-total-tokens MAX_TOTAL_TOKENS
                        The maximum number of tokens in the memory pool. If not specified, it will be
                        automatically calculated based on the memory usage fraction. This option is
                        typically used for development and debugging purposes.
  --chunked-prefill-size CHUNKED_PREFILL_SIZE
                        The maximum number of tokens in a chunk for the chunked prefill. Setting this to
                        -1 means disabling chunked prefill.
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        The maximum number of tokens in a prefill batch. The real bound will be the
                        maximum of this value and the model's maximum context length.
  --schedule-policy {lpm,random,fcfs,dfs-weight,lof}
                        The scheduling policy of the requests.
  --schedule-conservativeness SCHEDULE_CONSERVATIVENESS
                        How conservative the schedule policy is. A larger value means more conservative
                        scheduling. Use a larger value if you see requests being retracted frequently.
  --page-size PAGE_SIZE
                        The number of tokens in a page.
  --hybrid-kvcache-ratio [HYBRID_KVCACHE_RATIO]
                        Mix ratio in [0,1] between uniform and hybrid kv buffers (0.0 = pure uniform:
                        swa_size / full_size = 1)(1.0 = pure hybrid: swa_size / full_size =
                        local_attention_size / context_length)
  --swa-full-tokens-ratio SWA_FULL_TOKENS_RATIO
                        The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the
                        number of swa:full layers. It should be between 0 and 1. E.g. 0.5 means if each
                        swa layer has 50 tokens, then each full layer has 100 tokens.
  --disable-hybrid-swa-memory
                        Disable the hybrid SWA memory.
  --device DEVICE       The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-
                        detection if not specified.
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, --tp-size TENSOR_PARALLEL_SIZE
                        The tensor parallelism size.
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, --pp-size PIPELINE_PARALLEL_SIZE
                        The pipeline parallelism size.
  --max-micro-batch-size MAX_MICRO_BATCH_SIZE
                        The maximum micro batch size in pipeline parallelism.
  --stream-interval STREAM_INTERVAL
                        The interval (or buffer size) for streaming in terms of the token length. A
                        smaller value makes streaming smoother, while a larger value makes the
                        throughput higher
  --stream-output       Whether to output as a sequence of disjoint segments.
  --random-seed RANDOM_SEED
                        The random seed.
  --constrained-json-whitespace-pattern CONSTRAINED_JSON_WHITESPACE_PATTERN
                        (outlines backend only) Regex pattern for syntactic whitespaces allowed in JSON
                        constrained output. For example, to allow the model generate consecutive
                        whitespaces, set the pattern to [ ]*
  --watchdog-timeout WATCHDOG_TIMEOUT
                        Set watchdog timeout in seconds. If a forward batch takes longer than this, the
                        server will crash to prevent hanging.
  --dist-timeout DIST_TIMEOUT
                        Set timeout for torch.distributed initialization.
  --download-dir DOWNLOAD_DIR
                        Model download directory for huggingface.
  --base-gpu-id BASE_GPU_ID
                        The base GPU ID to start allocating GPUs from. Useful when running multiple
                        instances on the same machine.
  --gpu-id-step GPU_ID_STEP
                        The delta between consecutive GPU IDs that are used. For example, setting it to
                        2 will use GPU 0,2,4,...
  --sleep-on-idle       Reduce CPU usage when sglang is idle.
  --log-level LOG_LEVEL
                        The logging level of all loggers.
  --log-level-http LOG_LEVEL_HTTP
                        The logging level of HTTP server. If not set, reuse --log-level by default.
  --log-requests        Log metadata, inputs, outputs of all requests. The verbosity is decided by
                        --log-requests-level
  --log-requests-level {0,1,2,3}
                        0: Log metadata (no sampling parameters). 1: Log metadata and sampling
                        parameters. 2: Log metadata, sampling parameters and partial input/output. 3:
                        Log every input/output.
  --crash-dump-folder CRASH_DUMP_FOLDER
                        Folder path to dump requests from the last 5 min before a crash (if any). If not
                        specified, crash dumping is disabled.
  --show-time-cost      Show time cost of custom marks.
  --enable-metrics      Enable log prometheus metrics.
  --enable-metrics-for-all-schedulers
                        Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP
                        ranks (not just TP 0) to record request metrics separately. This is especially
                        useful when dp_attention is enabled, as otherwise all metrics appear to come
                        from TP 0.
  --bucket-time-to-first-token BUCKET_TIME_TO_FIRST_TOKEN [BUCKET_TIME_TO_FIRST_TOKEN ...]
                        The buckets of time to first token, specified as a list of floats.
  --bucket-inter-token-latency BUCKET_INTER_TOKEN_LATENCY [BUCKET_INTER_TOKEN_LATENCY ...]
                        The buckets of inter-token latency, specified as a list of floats.
  --bucket-e2e-request-latency BUCKET_E2E_REQUEST_LATENCY [BUCKET_E2E_REQUEST_LATENCY ...]
                        The buckets of end-to-end request latency, specified as a list of floats.
  --collect-tokens-histogram
                        Collect prompt/generation tokens histogram.
  --prompt-tokens-buckets PROMPT_TOKENS_BUCKETS [PROMPT_TOKENS_BUCKETS ...]
                        The buckets rule of prompt tokens. Supports 3 rule types: 'default' uses
                        predefined buckets; 'tse <middle> <base> <count>' generates two sides
                        exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0,
                        992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'customer
                        <value1> <value2> ...' uses custom bucket values (e.g., 'customer 10 50 100
                        500').
  --generation-tokens-buckets GENERATION_TOKENS_BUCKETS [GENERATION_TOKENS_BUCKETS ...]
                        The buckets rule for generation tokens histogram. Supports 3 rule types:
                        'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two
                        sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets
                        [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).);
                        'customer <value1> <value2> ...' uses custom bucket values (e.g., 'customer 10
                        50 100 500').
  --gc-warning-threshold-secs GC_WARNING_THRESHOLD_SECS
                        The threshold for long GC warning. If a GC takes longer than this, a warning
                        will be logged. Set to 0 to disable.
  --decode-log-interval DECODE_LOG_INTERVAL
                        The log interval of decode batch.
  --enable-request-time-stats-logging
                        Enable per request time stats logging
  --kv-events-config KV_EVENTS_CONFIG
                        Config in json format for NVIDIA dynamo KV event publishing. Publishing will be
                        enabled if this flag is used.
  --api-key API_KEY     Set API key of the server. It is also used in the OpenAI API compatible server.
  --served-model-name SERVED_MODEL_NAME
                        Override the model name returned by the v1/models endpoint in OpenAI API server.
  --weight-version WEIGHT_VERSION
                        Version identifier for the model weights. Defaults to 'default' if not
                        specified.
  --chat-template CHAT_TEMPLATE
                        The buliltin chat template name or the path of the chat template file. This is
                        only used for OpenAI-compatible API server.
  --completion-template COMPLETION_TEMPLATE
                        The buliltin completion template name or the path of the completion template
                        file. This is only used for OpenAI-compatible API server. only for code
                        completion currently.
  --file-storage-path FILE_STORAGE_PATH
                        The path of the file storage in backend.
  --enable-cache-report
                        Return number of cached tokens in usage.prompt_tokens_details for each openai
                        request.
  --reasoning-parser {deepseek-r1,deepseek-v3,glm45,gpt-oss,kimi,qwen3,qwen3-thinking,step3}
                        Specify the parser for reasoning models, supported parsers are: ['deepseek-r1',
                        'deepseek-v3', 'glm45', 'gpt-oss', 'kimi', 'qwen3', 'qwen3-thinking', 'step3'].
  --tool-call-parser {llama3,qwen25,mistral,deepseekv3,deepseekv31,pythonic,kimi_k2,qwen3_coder,glm45,step3,gpt-oss}
                        Specify the parser for handling tool-call interactions. Options include:
                        ['llama3', 'qwen25', 'mistral', 'deepseekv3', 'deepseekv31', 'pythonic',
                        'kimi_k2', 'qwen3_coder', 'glm45', 'step3', 'gpt-oss'].
  --tool-server TOOL_SERVER
                        Either 'demo' or a comma-separated list of tool server urls to use for the
                        model. If not specified, no tool server will be used.
  --data-parallel-size DATA_PARALLEL_SIZE, --dp-size DATA_PARALLEL_SIZE
                        The data parallelism size.
  --load-balance-method {round_robin,shortest_queue,minimum_tokens}
                        The load balancing strategy for data parallelism.
  --prefill-round-robin-balance
                        Prefill is round robin balanced. This is used to promise decode server can get
                        the correct dp rank.
  --dist-init-addr DIST_INIT_ADDR, --nccl-init-addr DIST_INIT_ADDR
                        The host address for initializing distributed backend (e.g.,
                        `192.168.0.2:25000`).
  --nnodes NNODES       The number of nodes.
  --node-rank NODE_RANK
                        The node rank.
  --json-model-override-args JSON_MODEL_OVERRIDE_ARGS
                        A dictionary in JSON string format used to override default model
                        configurations.
  --preferred-sampling-params PREFERRED_SAMPLING_PARAMS
                        json-formatted sampling settings that will be returned in /get_model_info
  --enable-lora         Enable LoRA support for the model. This argument is automatically set to True if
                        `--lora-paths` is provided for backward compatibility.
  --max-lora-rank MAX_LORA_RANK
                        The maximum rank of LoRA adapters. If not specified, it will be automatically
                        inferred from the adapters provided in --lora-paths.
  --lora-target-modules [{q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv_proj,gate_up_proj,all} ...]
                        The union set of all target modules where LoRA should be applied. If not
                        specified, it will be automatically inferred from the adapters provided in
                        --lora-paths. If 'all' is specified, all supported modules will be targeted.
  --lora-paths [LORA_PATHS ...]
                        The list of LoRA adapters to load. Each adapter must be specified in one of the
                        following formats: <PATH> | <NAME>=<PATH> | JSON with schema
                        {"lora_name":str,"lora_path":str,"pinned":bool}
  --max-loras-per-batch MAX_LORAS_PER_BATCH
                        Maximum number of adapters for a running batch, include base-only request.
  --max-loaded-loras MAX_LOADED_LORAS
                        If specified, it limits the maximum number of LoRA adapters loaded in CPU memory
                        at a time. The value must be greater than or equal to `--max-loras-per-batch`.
  --lora-backend LORA_BACKEND
                        Choose the kernel backend for multi-LoRA serving.
  --attention-backend {triton,torch_native,cutlass_mla,fa3,flashinfer,flashmla,trtllm_mla,trtllm_mha,dual_chunk_flash_attn,hybrid_linear_attn,aiter,wave,intel_amx,ascend}
                        Choose the kernels for attention layers.
  --prefill-attention-backend {triton,torch_native,cutlass_mla,fa3,flashinfer,flashmla,trtllm_mla,trtllm_mha,dual_chunk_flash_attn,hybrid_linear_attn,aiter,wave,intel_amx,ascend}
                        Choose the kernels for prefill attention layers (have priority over --attention-
                        backend).
  --decode-attention-backend {triton,torch_native,cutlass_mla,fa3,flashinfer,flashmla,trtllm_mla,trtllm_mha,dual_chunk_flash_attn,hybrid_linear_attn,aiter,wave,intel_amx,ascend}
                        Choose the kernels for decode attention layers (have priority over --attention-
                        backend).
  --sampling-backend {flashinfer,pytorch}
                        Choose the kernels for sampling layers.
  --grammar-backend {xgrammar,outlines,llguidance,none}
                        Choose the backend for grammar-guided decoding.
  --mm-attention-backend {sdpa,fa3,triton_attn}
                        Set multimodal attention backend.
  --speculative-algorithm {EAGLE,EAGLE3,NEXTN,STANDALONE}
                        Speculative algorithm.
  --speculative-draft-model-path SPECULATIVE_DRAFT_MODEL_PATH, --speculative-draft-model SPECULATIVE_DRAFT_MODEL_PATH
                        The path of the draft model weights. This can be a local folder or a Hugging
                        Face repo ID.
  --speculative-draft-model-revision SPECULATIVE_DRAFT_MODEL_REVISION
                        The specific draft model version to use. It can be a branch name, a tag name, or
                        a commit id. If unspecified, will use the default version.
  --speculative-num-steps SPECULATIVE_NUM_STEPS
                        The number of steps sampled from draft model in Speculative Decoding.
  --speculative-eagle-topk SPECULATIVE_EAGLE_TOPK
                        The number of tokens sampled from the draft model in eagle2 each step.
  --speculative-num-draft-tokens SPECULATIVE_NUM_DRAFT_TOKENS
                        The number of tokens sampled from the draft model in Speculative Decoding.
  --speculative-accept-threshold-single SPECULATIVE_ACCEPT_THRESHOLD_SINGLE
                        Accept a draft token if its probability in the target model is greater than this
                        threshold.
  --speculative-accept-threshold-acc SPECULATIVE_ACCEPT_THRESHOLD_ACC
                        The accept probability of a draft token is raised from its target probability p
                        to min(1, p / threshold_acc).
  --speculative-token-map SPECULATIVE_TOKEN_MAP
                        The path of the draft model's small vocab table.
  --speculative-attention-mode {prefill,decode}
                        Attention backend for speculative decoding operations (both target verify and
                        draft extend). Can be one of 'prefill' (default) or 'decode'.
  --expert-parallel-size EXPERT_PARALLEL_SIZE, --ep-size EXPERT_PARALLEL_SIZE, --ep EXPERT_PARALLEL_SIZE
                        The expert parallelism size.
  --moe-a2a-backend {none,deepep}
                        Choose the backend for MoE A2A.
  --moe-runner-backend {auto,triton,triton_kernel,flashinfer_trtllm,flashinfer_cutlass,flashinfer_mxfp4}
                        Choose the runner backend for MoE.
  --flashinfer-mxfp4-moe-precision {default,bf16}
                        Choose the computation precision of flashinfer mxfp4 moe
  --enable-flashinfer-allreduce-fusion
                        Enable FlashInfer allreduce fusion with Residual RMSNorm.
  --deepep-mode {normal,low_latency,auto}
                        Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or
                        `auto`. Default is `auto`, which means `low_latency` for decode batch and
                        `normal` for prefill batch.
  --ep-num-redundant-experts EP_NUM_REDUNDANT_EXPERTS
                        Allocate this number of redundant experts in expert parallel.
  --ep-dispatch-algorithm EP_DISPATCH_ALGORITHM
                        The algorithm to choose ranks for redundant experts in expert parallel.
  --init-expert-location INIT_EXPERT_LOCATION
                        Initial location of EP experts.
  --enable-eplb         Enable EPLB algorithm
  --eplb-algorithm EPLB_ALGORITHM
                        Chosen EPLB algorithm
  --eplb-rebalance-num-iterations EPLB_REBALANCE_NUM_ITERATIONS
                        Number of iterations to automatically trigger a EPLB re-balance.
  --eplb-rebalance-layers-per-chunk EPLB_REBALANCE_LAYERS_PER_CHUNK
                        Number of layers to rebalance per forward pass.
  --eplb-min-rebalancing-utilization-threshold EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD
                        Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must
                        be in the range [0.0, 1.0].
  --expert-distribution-recorder-mode EXPERT_DISTRIBUTION_RECORDER_MODE
                        Mode of expert distribution recorder.
  --expert-distribution-recorder-buffer-size EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE
                        Circular buffer size of expert distribution recorder. Set to -1 to denote
                        infinite buffer.
  --enable-expert-distribution-metrics
                        Enable logging metrics for expert balancedness
  --deepep-config DEEPEP_CONFIG
                        Tuned DeepEP config suitable for your own cluster. It can be either a string
                        with JSON content or a file path.
  --moe-dense-tp-size MOE_DENSE_TP_SIZE
                        TP size for MoE dense MLP layers. This flag is useful when, with large TP size,
                        there are errors caused by weights in MLP layers having dimension smaller than
                        the min dimension GEMM supports.
  --max-mamba-cache-size MAX_MAMBA_CACHE_SIZE
                        The maximum size of the mamba cache.
  --mamba-ssm-dtype {float32,bfloat16}
                        The data type of the SSM states in mamba cache.
  --enable-hierarchical-cache
                        Enable hierarchical cache
  --hicache-ratio HICACHE_RATIO
                        The ratio of the size of host KV cache memory pool to the size of device pool.
  --hicache-size HICACHE_SIZE
                        The size of host KV cache memory pool in gigabytes, which will override the
                        hicache_ratio if set.
  --hicache-write-policy {write_back,write_through,write_through_selective}
                        The write policy of hierarchical cache.
  --hicache-io-backend {direct,kernel}
                        The IO backend for KV cache transfer between CPU and GPU
  --hicache-mem-layout {layer_first,page_first}
                        The layout of host memory pool for hierarchical cache.
  --hicache-storage-backend {file,mooncake,hf3fs,nixl}
                        The storage backend for hierarchical KV cache.
  --hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}
                        Control when prefetching from the storage backend should stop.
  --hicache-storage-backend-extra-config HICACHE_STORAGE_BACKEND_EXTRA_CONFIG
                        A dictionary in JSON string format containing extra configuration for the
                        storage backend.
  --enable-lmcache      Using LMCache as an alternative hierarchical cache solution
  --enable-double-sparsity
                        Enable double sparsity attention
  --ds-channel-config-path DS_CHANNEL_CONFIG_PATH
                        The path of the double sparsity channel config
  --ds-heavy-channel-num DS_HEAVY_CHANNEL_NUM
                        The number of heavy channels in double sparsity attention
  --ds-heavy-token-num DS_HEAVY_TOKEN_NUM
                        The number of heavy tokens in double sparsity attention
  --ds-heavy-channel-type DS_HEAVY_CHANNEL_TYPE
                        The type of heavy channels in double sparsity attention
  --ds-sparse-decode-threshold DS_SPARSE_DECODE_THRESHOLD
                        The type of heavy channels in double sparsity attention
  --cpu-offload-gb CPU_OFFLOAD_GB
                        How many GBs of RAM to reserve for CPU offloading.
  --offload-group-size OFFLOAD_GROUP_SIZE
                        Number of layers per group in offloading.
  --offload-num-in-group OFFLOAD_NUM_IN_GROUP
                        Number of layers to be offloaded within a group.
  --offload-prefetch-step OFFLOAD_PREFETCH_STEP
                        Steps to prefetch in offloading.
  --offload-mode OFFLOAD_MODE
                        Mode of offloading.
  --disable-radix-cache
                        Disable RadixAttention for prefix caching.
  --cuda-graph-max-bs CUDA_GRAPH_MAX_BS
                        Set the maximum batch size for cuda graph. It will extend the cuda graph capture
                        batch size to this value.
  --cuda-graph-bs CUDA_GRAPH_BS [CUDA_GRAPH_BS ...]
                        Set the list of batch sizes for cuda graph.
  --disable-cuda-graph  Disable cuda graph.
  --disable-cuda-graph-padding
                        Disable cuda graph when padding is needed. Still uses cuda graph when padding is
                        not needed.
  --enable-profile-cuda-graph
                        Enable profiling of cuda graph capture.
  --enable-cudagraph-gc
                        Enable garbage collection during CUDA graph capture. If disabled (default), GC
                        is frozen during capture to speed up the process.
  --enable-nccl-nvls    Enable NCCL NVLS for prefill heavy requests when available.
  --enable-symm-mem     Enable NCCL symmetric memory for fast collectives.
  --disable-flashinfer-cutlass-moe-fp4-allgather
                        Disables quantize before all-gather for flashinfer cutlass moe.
  --enable-tokenizer-batch-encode
                        Enable batch tokenization for improved performance when processing multiple text
                        inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.
  --disable-outlines-disk-cache
                        Disable disk cache of outlines to avoid possible crashes related to file system
                        or high concurrency.
  --disable-custom-all-reduce
                        Disable the custom all-reduce kernel and fall back to NCCL.
  --enable-mscclpp      Enable using mscclpp for small messages for all-reduce kernel and fall back to
                        NCCL.
  --disable-overlap-schedule
                        Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model
                        worker.
  --enable-mixed-chunk  Enabling mixing prefill and decode in a batch when using chunked prefill.
  --enable-dp-attention
                        Enabling data parallelism for attention and tensor parallelism for FFN. The dp
                        size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE
                        models are supported.
  --enable-dp-lm-head   Enable vocabulary parallel across the attention TP group to avoid all-gather
                        across DP groups, optimizing performance under DP attention.
  --enable-two-batch-overlap
                        Enabling two micro batches to overlap.
  --tbo-token-distribution-threshold TBO_TOKEN_DISTRIBUTION_THRESHOLD
                        The threshold of token distribution between two batches in micro-batch-overlap,
                        determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote
                        disable two-chunk-overlap.
  --enable-torch-compile
                        Optimize the model with torch.compile. Experimental feature.
  --torch-compile-max-bs TORCH_COMPILE_MAX_BS
                        Set the maximum batch size when using torch compile.
  --torchao-config TORCHAO_CONFIG
                        Optimize the model with torchao. Experimental feature. Current choices are:
                        int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row
  --enable-nan-detection
                        Enable the NaN detection for debugging purposes.
  --enable-p2p-check    Enable P2P check for GPU access, otherwise the p2p access is allowed by default.
  --triton-attention-reduce-in-fp32
                        Cast the intermediate attention results to fp32 to avoid possible crashes
                        related to fp16.This only affects Triton attention kernels.
  --triton-attention-num-kv-splits TRITON_ATTENTION_NUM_KV_SPLITS
                        The number of KV splits in flash decoding Triton kernel. Larger value is better
                        in longer context scenarios. The default value is 8.
  --num-continuous-decode-steps NUM_CONTINUOUS_DECODE_STEPS
                        Run multiple continuous decoding steps to reduce scheduling overhead. This can
                        potentially increase throughput but may also increase time-to-first-token
                        latency. The default value is 1, meaning only run one decoding step at a time.
  --delete-ckpt-after-loading
                        Delete the model checkpoint after loading the model.
  --enable-memory-saver
                        Allow saving memory using release_memory_occupation and resume_memory_occupation
  --allow-auto-truncate
                        Allow automatically truncating requests that exceed the maximum input length
                        instead of returning an error.
  --enable-custom-logit-processor
                        Enable users to pass custom logit processors to the server (disabled by default
                        for security)
  --flashinfer-mla-disable-ragged
                        Not using ragged prefill wrapper when running flashinfer mla
  --disable-shared-experts-fusion
                        Disable shared experts fusion optimization for deepseek v3/r1.
  --disable-chunked-prefix-cache
                        Disable chunked prefix cache feature for deepseek, which should save overhead
                        for short sequences.
  --disable-fast-image-processor
                        Adopt base image processor instead of fast image processor.
  --enable-return-hidden-states
                        Enable returning hidden states with responses.
  --scheduler-recv-interval SCHEDULER_RECV_INTERVAL
                        The interval to poll requests in scheduler. Can be set to >1 to reduce the
                        overhead of this.
  --numa-node NUMA_NODE [NUMA_NODE ...]
                        Sets the numa node for the subprocesses. i-th element corresponds to i-th
                        subprocess.
  --debug-tensor-dump-output-folder DEBUG_TENSOR_DUMP_OUTPUT_FOLDER
                        The output folder for dumping tensors.
  --debug-tensor-dump-input-file DEBUG_TENSOR_DUMP_INPUT_FILE
                        The input filename for dumping tensors
  --debug-tensor-dump-inject DEBUG_TENSOR_DUMP_INJECT
                        Inject the outputs from jax as the input of every layer.
  --debug-tensor-dump-prefill-only
                        Only dump the tensors for prefill requests (i.e. batch size > 1).
  --disaggregation-mode {null,prefill,decode}
                        Only used for PD disaggregation. "prefill" for prefill-only server, and "decode"
                        for decode-only server. If not specified, it is not PD disaggregated
  --disaggregation-transfer-backend {mooncake,nixl,ascend,fake}
                        The backend for disaggregation transfer. Default is mooncake.
  --disaggregation-bootstrap-port DISAGGREGATION_BOOTSTRAP_PORT
                        Bootstrap server port on the prefill server. Default is 8998.
  --disaggregation-decode-tp DISAGGREGATION_DECODE_TP
                        Decode tp size. If not set, it matches the tp size of the current engine. This
                        is only set on the prefill server.
  --disaggregation-decode-dp DISAGGREGATION_DECODE_DP
                        Decode dp size. If not set, it matches the dp size of the current engine. This
                        is only set on the prefill server.
  --disaggregation-prefill-pp DISAGGREGATION_PREFILL_PP
                        Prefill pp size. If not set, it is default to 1. This is only set on the decode
                        server.
  --disaggregation-ib-device DISAGGREGATION_IB_DEVICE
                        The InfiniBand devices for disaggregation transfer, accepts single device (e.g.,
                        --disaggregation-ib-device mlx5_0) or multiple comma-separated devices (e.g.,
                        --disaggregation-ib-device mlx5_0,mlx5_1). Default is None, which triggers
                        automatic device detection when mooncake backend is enabled.
  --num-reserved-decode-tokens NUM_RESERVED_DECODE_TOKENS
                        Number of decode tokens that will have memory reserved when adding new request
                        to the running batch.
  --custom-weight-loader [CUSTOM_WEIGHT_LOADER ...]
                        The custom dataloader which used to update the model. Should be set with a valid
                        import path, such as my_package.weight_load_func
  --weight-loader-disable-mmap
                        Disable mmap while loading weight using safetensors.
  --enable-pdmux        Enable PD-Multiplexing, PD running on greenctx stream.
  --sm-group-num SM_GROUP_NUM
                        Number of sm partition groups.
  --enable-ep-moe       (Deprecated) Enabling expert parallelism for moe. The ep size is equal to the tp
                        size.
  --enable-deepep-moe   (Deprecated) Enabling DeepEP MoE implementation for EP MoE.
  --enable-flashinfer-cutlass-moe
                        (Deprecated) Enable FlashInfer CUTLASS MoE backend for modelopt_fp4 quant on
                        Blackwell. Supports MoE-EP
  --enable-flashinfer-trtllm-moe
                        (Deprecated) Enable FlashInfer TRTLLM MoE backend on Blackwell. Supports
                        BlockScale FP8 MoE-EP
  --enable-triton-kernel-moe
                        (Deprecated) Use triton moe grouped gemm kernel.
  --enable-flashinfer-mxfp4-moe
                        (Deprecated) Enable FlashInfer MXFP4 MoE backend for modelopt_fp4 quant on
                        Blackwell.
  --backend BACKEND
  --result-filename RESULT_FILENAME
  --dataset-name {sharegpt,random,generated-shared-prefix}
                        Name of the dataset to benchmark on.
  --dataset-path DATASET_PATH
                        Path to the dataset.
  --num-prompts NUM_PROMPTS
                        Number of prompts to process. Default is 1000.
  --sharegpt-output-len SHAREGPT_OUTPUT_LEN
                        Output length for each request. Overrides the output length from the ShareGPT
                        dataset.
  --sharegpt-context-len SHAREGPT_CONTEXT_LEN
                        The context length of the model for the ShareGPT dataset. Requests longer than
                        the context length will be dropped.
  --random-input-len RANDOM_INPUT_LEN
                        Number of input tokens per request, used only for random dataset.
  --random-output-len RANDOM_OUTPUT_LEN
                        Number of output tokens per request, used only for random dataset.
  --random-range-ratio RANDOM_RANGE_RATIO
                        Range of sampled ratio of input/output length, used only for random dataset.
  --gsp-num-groups GSP_NUM_GROUPS
                        Number of groups with shared prefix, usedonly for generate-shared-prefix
  --gsp-prompts-per-group GSP_PROMPTS_PER_GROUP
                        Number of prompts per group of shared prefix, usedonly for generate-shared-
                        prefix
  --gsp-system-prompt-len GSP_SYSTEM_PROMPT_LEN
                        System prompt length, usedonly for generate-shared-prefix
  --gsp-question-len GSP_QUESTION_LEN
                        Question length, usedonly for generate-shared-prefix
  --gsp-output-len GSP_OUTPUT_LEN
                        Target length in tokens for outputs in generated-shared-prefix dataset
  --seed SEED           The random seed.
  --disable-ignore-eos  Disable ignore EOS token
  --extra-request-body {"key1": "value1", "key2": "value2"}
                        Append given JSON object to the request payload. You can use this to
                        specifyadditional generate params like sampling params.
  --apply-chat-template
                        Apply chat template
  --profile             Use Torch Profiler. The endpoint must be launched with SGLANG_TORCH_PROFILER_DIR
                        to enable profiler.
  --skip-warmup         Skip the warmup batches.
  --do-not-exit         Do not exit the program. This is useful for nsys profile with --duration and
                        --delay.
  --prompt-suffix PROMPT_SUFFIX
                        Suffix applied to the end of all user prompts, followed by assistant prompt
                        suffix.