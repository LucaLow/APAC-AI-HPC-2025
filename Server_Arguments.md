# Extracted from [sglang docs](https://docs.sglang.ai/advanced_features/server_arguments.html)
# Server Arguments

This page provides a list of server arguments used on the command line to configure the behavior and performance of the language model server during deployment. These arguments let you customize model selection, parallelism, memory management, and optimizations.

You can list all arguments with:

```bash
python3 -m sglang.launch_server --help
```

---

## Common launch commands

* **Enable multi-GPU tensor parallelism** (add `--tp 2`).
  If you see “peer access is not supported between these two devices”, add `--enable-p2p-check`.

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --tp 2
```

* **Enable multi-GPU data parallelism** (add `--dp 2`). DP is better for throughput if there’s enough memory. It can be used together with TP. The following uses 4 GPUs in total. (Recommend using **SGLang Router** for DP.)

```bash
python -m sglang_router.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --dp 2 --tp 2
```

* **Reduce KV-cache pool memory** if you hit OOM during serving (set a smaller `--mem-fraction-static`; default is `0.9`):

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --mem-fraction-static 0.7
```

* See **[hyperparameter tuning](hyperparameter_tuning.html)** for performance tuning.

* For **Docker/Kubernetes**, set up shared memory for inter-process communication (`--shm-size` in Docker; update `/dev/shm` in K8s).

* **Long prompts prefill OOM**: try a smaller chunked prefill size.

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --chunked-prefill-size 4096
```

* **Enable `torch.compile` acceleration**:

  Add `--enable-torch-compile`. Cache directory defaults to `/tmp/torchinductor_root`; customize via `TORCHINDUCTOR_CACHE_DIR`. See:

  * PyTorch docs: [https://pytorch.org/tutorials/recipes/torch\_compile\_caching\_tutorial.html](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
  * SGLang docs: *Enabling cache for torch.compile* (in `backend/hyperparameter_tuning.html`)

* **Enable torchao quantization**:

```bash
--torchao-config int4wo-128
```

(Supports other strategies like INT8/FP8.)

* **Enable FP8 weight quantization**:

```bash
--quantization fp8
```

(Or load an FP8 checkpoint directly.)

* **Enable FP8 KV-cache quantization**:

```bash
--kv-cache-dtype fp8_e5m2
```

* **Custom chat template** if the tokenizer lacks one:

```bash
--chat-template <path_or_builtin_name>
```

* **Tensor parallelism across multiple nodes**:

```bash
# Node 0
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --tp 4 --dist-init-addr sgl-dev-0:50000 \
  --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --tp 4 --dist-init-addr sgl-dev-0:50000 \
  --nnodes 2 --node-rank 1
```

If you meet a deadlock, try `--disable-cuda-graph`.

Consult the sections below and [`server_args.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) for more.

---

## Model and tokenizer

| Argument                | Description                                                                                        | Default |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ------- |
| `--model-path`          | Path to model weights (local folder or HF repo ID).                                                | —       |
| `--tokenizer-path`      | Path to tokenizer.                                                                                 | —       |
| `--tokenizer-mode`      | `auto` uses fast tokenizer if available; `slow` always uses slow.                                  | `auto`  |
| `--skip-tokenizer-init` | Skip tokenizer init and pass `input_ids` in generate request.                                      | `False` |
| `--load-format`         | Weight format: `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `gguf`, `bitsandbytes`, `layered`. | `auto`  |
| `--trust-remote-code`   | Allow custom modeling files from Hub.                                                              | `False` |
| `--context-length`      | Override model’s max context length.                                                               | `None`  |
| `--is-embedding`        | Use CausalLM as embedding model.                                                                   | `False` |
| `--enable-multimodal`   | Enable multimodal functionality.                                                                   | `None`  |
| `--revision`            | Specific model version (branch/tag/commit).                                                        | `None`  |
| `--model-impl`          | `auto`/`sglang`/`transformers`.                                                                    | `auto`  |

---

## HTTP server

| Argument               | Description            | Default     |
| ---------------------- | ---------------------- | ----------- |
| `--host`               | Server host.           | `127.0.0.1` |
| `--port`               | Server port.           | `30000`     |
| `--skip-server-warmup` | Skip warmup.           | `False`     |
| `--warmups`            | Warmup configurations. | `None`      |
| `--nccl-port`          | Port for NCCL init.    | `None`      |

---

## Quantization and data type

| Argument                    | Description                                                                                    | Default |
| --------------------------- | ---------------------------------------------------------------------------------------------- | ------- |
| `--dtype`                   | Data type for weights/activations (`auto`, `half`, `float16`, `bfloat16`, `float`, `float32`). | `auto`  |
| `--quantization`            | Quantization method.                                                                           | `None`  |
| `--quantization-param-path` | JSON path for KV-cache scaling factors (esp. FP8 KV).                                          | `None`  |
| `--kv-cache-dtype`          | KV-cache dtype (`auto`, `fp8_e5m2`, `fp8_e4m3`).                                               | `auto`  |

---

## Memory and scheduling

| Argument                      | Description                                               | Default |
| ----------------------------- | --------------------------------------------------------- | ------- |
| `--mem-fraction-static`       | Fraction for static allocation (weights + KV pool).       | `None`  |
| `--max-running-requests`      | Max concurrent running requests.                          | `None`  |
| `--max-total-tokens`          | Max tokens in memory pool.                                | `None`  |
| `--chunked-prefill-size`      | Max tokens per chunk in chunked prefill (`-1` disables).  | `None`  |
| `--max-prefill-tokens`        | Max tokens in a prefill batch (bounded by model context). | `16384` |
| `--schedule-policy`           | Request scheduling policy.                                | `fcfs`  |
| `--schedule-conservativeness` | Larger ⇒ more conservative scheduling.                    | `1.0`   |
| `--cpu-offload-gb`            | GB of RAM reserved for CPU offload.                       | `0`     |
| `--page-size`                 | Tokens per page.                                          | `1`     |

---

## Runtime options

| Argument                                | Description                                                  | Default |
| --------------------------------------- | ------------------------------------------------------------ | ------- |
| `--device`                              | `cuda`, `xpu`, `hpu`, `npu`, `cpu` (auto-detect if omitted). | `None`  |
| `--tp-size`                             | Tensor parallel size.                                        | `1`     |
| `--pp-size`                             | Pipeline parallel size.                                      | `1`     |
| `--max-micro-batch-size`                | Max micro-batch size for PP.                                 | `None`  |
| `--stream-interval`                     | Streaming interval (token length).                           | `1`     |
| `--stream-output`                       | Output as disjoint segments.                                 | `False` |
| `--random-seed`                         | Random seed.                                                 | `None`  |
| `--constrained-json-whitespace-pattern` | Regex for allowed JSON whitespaces.                          | `None`  |
| `--watchdog-timeout`                    | Crash if a forward batch exceeds this (sec).                 | `300`   |
| `--dist-timeout`                        | Timeout for torch.distributed init.                          | `None`  |
| `--download-dir`                        | Model download dir (HF).                                     | `None`  |
| `--base-gpu-id`                         | Base GPU ID when allocating GPUs.                            | `0`     |
| `--gpu-id-step`                         | Step between allocated GPU IDs (e.g., `0,2,4,...`).          | `1`     |
| `--sleep-on-idle`                       | Reduce CPU usage when idle.                                  | `False` |

---

## Logging

| Argument                              | Description                                        | Default |
| ------------------------------------- | -------------------------------------------------- | ------- |
| `--log-level`                         | Logging level for all loggers.                     | `info`  |
| `--log-level-http`                    | HTTP server log level (defaults to `--log-level`). | `None`  |
| `--log-requests`                      | Log metadata/inputs/outputs per request.           | `False` |
| `--log-requests-level`                | 0..3 verbosity of request logging.                 | `0`     |
| `--show-time-cost`                    | Show custom mark timing.                           | `False` |
| `--enable-metrics`                    | Enable Prometheus metrics.                         | `False` |
| `--bucket-time-to-first-token`        | Buckets for TTFT.                                  | `None`  |
| `--bucket-inter-token-latency`        | Buckets for inter-token latency.                   | `None`  |
| `--bucket-e2e-request-latency`        | Buckets for E2E request latency.                   | `None`  |
| `--collect-tokens-histogram`          | Collect prompt/generation token histograms.        | `False` |
| `--kv-events-config`                  | NVIDIA dynamo KV event publishing config (JSON).   | `None`  |
| `--decode-log-interval`               | Decode batch log interval.                         | `40`    |
| `--enable-request-time-stats-logging` | Per-request timing stats.                          | `False` |

---

## API related

| Argument                | Description                                                                | Default          |
| ----------------------- | -------------------------------------------------------------------------- | ---------------- |
| `--api-key`             | Server API key (also used by OpenAI-compatible server).                    | `None`           |
| `--served-model-name`   | Override name returned by `v1/models`.                                     | `None`           |
| `--chat-template`       | Built-in or path to chat template (OpenAI-compat server).                  | `None`           |
| `--completion-template` | Built-in or path to completion template (code completion).                 | `None`           |
| `--file-storage-path`   | Backend file storage path.                                                 | `sglang_storage` |
| `--enable-cache-report` | Return cached token counts in `usage.prompt_tokens_details`.               | `False`          |
| `--reasoning-parser`    | Parser for reasoning models.                                               | `None`           |
| `--tool-call-parser`    | Parser for tool-calls (`qwen25`, `mistral`, `llama3`, `deepseekv3`, etc.). | `None`           |

---

## Data parallelism

| Argument                | Description                                                                            | Default       |
| ----------------------- | -------------------------------------------------------------------------------------- | ------------- |
| `--dp-size`             | Data parallel size.                                                                    | `1`           |
| `--load-balance-method` | DP load balancing: `round_robin`, `minimum_tokens` (the latter requires DP attention). | `round_robin` |

---

## Multi-node distributed serving

| Argument           | Description                                                  | Default |
| ------------------ | ------------------------------------------------------------ | ------- |
| `--dist-init-addr` | Host\:port for distributed init (e.g., `192.168.0.2:25000`). | `None`  |
| `--nnodes`         | Number of nodes.                                             | `1`     |
| `--node-rank`      | Node rank.                                                   | `0`     |

---

## Model override args in JSON

| Argument                      | Description                                           | Default |
| ----------------------------- | ----------------------------------------------------- | ------- |
| `--json-model-override-args`  | JSON dict to override default model configs.          | `{}`    |
| `--preferred-sampling-params` | JSON sampling settings returned by `/get_model_info`. | `None`  |

---

## LoRA

| Argument                | Description                                                          | Default  |
| ----------------------- | -------------------------------------------------------------------- | -------- |
| `--enable-lora`         | Enable LoRA (auto-enabled if `--lora-paths` is provided).            | `False`  |
| `--max-lora-rank`       | Max LoRA rank to support.                                            | `None`   |
| `--lora-target-modules` | Union of target modules (`q_proj`, `k_proj`, `gate_proj`, or `all`). | `None`   |
| `--lora-paths`          | List of LoRA adapters to load.                                       | —        |
| `--max-loras-per-batch` | Max adapters per running batch (incl. base-only).                    | `8`      |
| `--max-loaded-loras`    | Limit max adapters loaded in CPU memory. ≥ `--max-loras-per-batch`.  | `None`   |
| `--lora-backend`        | Kernel backend for multi-LoRA.                                       | `triton` |

---

## Kernel backend

| Argument                      | Description                                                    | Default |
| ----------------------------- | -------------------------------------------------------------- | ------- |
| `--attention-backend`         | Kernel backend for attention.                                  | `None`  |
| `--prefill-attention-backend` | Backend for prefill attention (overrides `attention_backend`). | `None`  |
| `--decode-attention-backend`  | Backend for decode attention (overrides `attention_backend`).  | `None`  |
| `--sampling-backend`          | Kernel backend for sampling.                                   | `None`  |
| `--grammar-backend`           | Backend for grammar-guided decoding.                           | `None`  |
| `--mm-attention-backend`      | Backend for multimodal attention.                              | `None`  |

---

## Speculative decoding

| Argument                                | Description                                      | Default |
| --------------------------------------- | ------------------------------------------------ | ------- |
| `--speculative-algorithm`               | Speculative algorithm.                           | `None`  |
| `--speculative-draft-model-path`        | Path or HF ID of draft model.                    | `None`  |
| `--speculative-num-steps`               | Steps sampled from draft per round.              | `None`  |
| `--speculative-eagle-topk`              | `eagle2`: tokens sampled from draft each step.   | `None`  |
| `--speculative-num-draft-tokens`        | Number of draft tokens sampled.                  | `None`  |
| `--speculative-accept-threshold-single` | Accept token if target prob ≥ threshold.         | `1.0`   |
| `--speculative-accept-threshold-acc`    | Accept prob raised to `min(1, p/threshold_acc)`. | `1.0`   |
| `--speculative-token-map`               | Path to draft model’s small vocab table.         | `None`  |

---

## Expert parallelism (MoE)

| Argument                                     | Description                                                        | Default   |
| -------------------------------------------- | ------------------------------------------------------------------ | --------- |
| `--ep-size`                                  | Expert parallelism size.                                           | `1`       |
| `--moe-a2a-backend`                          | All-to-all backend for MoE EP.                                     | `none`    |
| `--moe-runner-backend`                       | Runner backend for MoE.                                            | `triton`  |
| `--deepep-mode`                              | DeepEP mode: `normal`, `low_latency`, `auto`.                      | `auto`    |
| `--ep-num-redundant-experts`                 | Number of redundant experts.                                       | `0`       |
| `--ep-dispatch-algorithm`                    | Rank selection algorithm for redundant experts (EPLB).             | `None`    |
| `--init-expert-location`                     | Initial EP expert placement.                                       | `trivial` |
| `--enable-eplb`                              | Enable EPLB (expert load balancing).                               | `False`   |
| `--eplb-algorithm`                           | EPLB algorithm.                                                    | `auto`    |
| `--eplb-rebalance-num-iterations`            | Iterations before auto re-balance.                                 | `1000`    |
| `--eplb-rebalance-layers-per-chunk`          | Layers to rebalance per forward.                                   | `None`    |
| `--expert-distribution-recorder-mode`        | Expert distribution recorder mode.                                 | `None`    |
| `--expert-distribution-recorder-buffer-size` | Recorder circular buffer size (`-1` = infinite).                   | `None`    |
| `--enable-expert-distribution-metrics`       | Log expert balancedness metrics.                                   | `False`   |
| `--deepep-config`                            | Tuned DeepEP config (JSON string or file path).                    | `None`    |
| `--moe-dense-tp-size`                        | TP size for MoE dense MLP layers (workaround for small GEMM dims). | `None`    |

---

## Hierarchical cache

| Argument                      | Description                           | Default         |
| ----------------------------- | ------------------------------------- | --------------- |
| `--enable-hierarchical-cache` | Enable hierarchical cache.            | `False`         |
| `--hicache-ratio`             | Host KV pool size / device pool size. | `2.0`           |
| `--hicache-size`              | Absolute HiCache size.                | `0`             |
| `--hicache-write-policy`      | Write policy (`write_through`, etc.). | `write_through` |
| `--hicache-io-backend`        | IO backend.                           | —               |
| `--hicache-storage-backend`   | Storage backend.                      | `None`          |

---

## Optimization / debug options

| Argument                             | Description                                                                                          | Default |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------- | ------- |
| `--disable-radix-cache`              | Disable RadixAttention prefix caching.                                                               | `False` |
| `--cuda-graph-max-bs`                | Max batch size for CUDA graph capture.                                                               | `None`  |
| `--cuda-graph-bs`                    | List of batch sizes for CUDA graph.                                                                  | `None`  |
| `--disable-cuda-graph`               | Disable CUDA graph.                                                                                  | `False` |
| `--disable-cuda-graph-padding`       | Disable CUDA graph when padding needed.                                                              | `False` |
| `--enable-profile-cuda-graph`        | Profile CUDA graph capture.                                                                          | `False` |
| `--enable-nccl-nvls`                 | Enable NCCL NVLS (prefill-heavy requests).                                                           | `False` |
| `--enable-symm-mem`                  | Enable NCCL symmetric memory.                                                                        | `False` |
| `--enable-tokenizer-batch-encode`    | Batch tokenization (avoid with images/pre-tokenized/embeds).                                         | `False` |
| `--disable-outlines-disk-cache`      | Disable outlines disk cache.                                                                         | `False` |
| `--disable-custom-all-reduce`        | Fall back to NCCL for all-reduce.                                                                    | `False` |
| `--enable-mscclpp`                   | Use mscclpp for small all-reduce messages.                                                           | `False` |
| `--disable-overlap-schedule`         | Disable CPU scheduler ↔ GPU worker overlap.                                                          | `False` |
| `--enable-mixed-chunk`               | Mix prefill & decode in chunked prefill batches.                                                     | `False` |
| `--enable-dp-attention`              | DP for attention + TP for FFN (`dp == tp`; DeepSeek-V2, Qwen 2/3 MoE supported).                     | `False` |
| `--enable-dp-lm-head`                | Vocab parallel across attention TP group to avoid DP all-gather.                                     | `False` |
| `--enable-two-batch-overlap`         | Overlap two micro-batches.                                                                           | `False` |
| `--tbo-token-distribution-threshold` | Threshold for two-batch vs two-chunk overlap.                                                        | `0.48`  |
| `--enable-torch-compile`             | Use `torch.compile` (experimental).                                                                  | `False` |
| `--torch-compile-max-bs`             | Max batch size under `torch.compile`.                                                                | `32`    |
| `--torchao-config`                   | TorchAO configs: `int8dq`, `int8wo`, `int4wo-<group>`, `fp8wo`, `fp8dq-per_tensor`, `fp8dq-per_row`. | —       |
| `--enable-nan-detection`             | Enable NaN detection (debug).                                                                        | `False` |
| `--enable-p2p-check`                 | Enforce P2P check for GPU access.                                                                    | `False` |
| `--triton-attention-reduce-in-fp32`  | Accumulate attention in fp32 (stability).                                                            | `False` |
| `--triton-attention-num-kv-splits`   | KV splits in flash-decode Triton kernel.                                                             | `8`     |
| `--num-continuous-decode-steps`      | Continuous decode steps per schedule (TTFT vs throughput trade-off).                                 | `1`     |
| `--delete-ckpt-after-loading`        | Delete checkpoint after loading.                                                                     | `False` |
| `--enable-memory-saver`              | Allow `release_memory_occupation` / `resume_memory_occupation`.                                      | `False` |
| `--allow-auto-truncate`              | Auto-truncate too-long requests instead of error.                                                    | `False` |
| `--enable-custom-logit-processor`    | Allow custom logit processors (security off by default).                                             | `False` |
| `--flashinfer-mla-disable-ragged`    | Disable ragged processing in FlashInfer MLA.                                                         | `False` |
| `--disable-shared-experts-fusion`    | Disable shared experts fusion.                                                                       | `False` |
| `--disable-chunked-prefix-cache`     | Disable chunked prefix cache.                                                                        | `False` |
| `--disable-fast-image-processor`     | Disable fast image processor.                                                                        | `False` |
| `--enable-return-hidden-states`      | Return hidden states.                                                                                | `False` |

---

## Debug tensor dumps

| Argument                            | Description                | Default |
| ----------------------------------- | -------------------------- | ------- |
| `--debug-tensor-dump-output-folder` | Output folder.             | `None`  |
| `--debug-tensor-dump-input-file`    | Input file.                | `None`  |
| `--debug-tensor-dump-inject`        | Inject debug tensor dumps. | `False` |
| `--debug-tensor-dump-prefill-only`  | Prefill-only mode.         | `False` |

---

## PD disaggregation

| Argument                            | Description                       | Default    |
| ----------------------------------- | --------------------------------- | ---------- |
| `--disaggregation-mode`             | `null`, `prefill`, or `decode`.   | `null`     |
| `--disaggregation-transfer-backend` | Transfer backend.                 | `mooncake` |
| `--disaggregation-bootstrap-port`   | Bootstrap port.                   | `8998`     |
| `--disaggregation-decode-tp`        | Decode TP for PD disaggregation.  | `None`     |
| `--disaggregation-decode-dp`        | Decode DP for PD disaggregation.  | `None`     |
| `--disaggregation-prefill-pp`       | Prefill PP for PD disaggregation. | `1`        |

---

## Model weight update

| Argument                       | Description                       | Default |
| ------------------------------ | --------------------------------- | ------- |
| `--custom-weight-loader`       | Custom weight loader paths.       | `None`  |
| `--weight-loader-disable-mmap` | Disable `mmap` for weight loader. | `False` |

---

## PD-Multiplexing

| Argument         | Description                              | Default |
| ---------------- | ---------------------------------------- | ------- |
| `--enable-pdmux` | Enable PD-Multiplexing.                  | `False` |
| `--sm-group-num` | Number of SM groups for PD-Multiplexing. | `3`     |
