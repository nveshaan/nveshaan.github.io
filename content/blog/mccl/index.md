---
title: "Distributed Training on Apple Silicon via MCCL"
date: 2026-06-20
draft: false
tags: ["deep learning", "pytorch", "mac"]
---

## Overview

**Collective Communication Libraries** such as `nccl` for nvidia gpus, `rccl` for amd gpus and `xccl` for intel gpus have enabled to scale up training of deep learning models to multiple gpu clusters. For apple gpus however, there exists `jaccl` but is exclusively for the **MLX Framework**. There was no way to use **PyTorch DDP** on Apple Silicon, up until the developers at [mps-ddp](https://github.com/mps-ddp) had created and graciously open-sourced [`mccl`](https://github.com/mps-ddp/mccl). The rest of the blog is a guide to do distributed training with PyTorch DDP on MCCL. All you need are two Macs and a Thunderbolt 4/5 cable.

{{< figure
    src="mccl.png"
    alt="mac setup"
    caption="2 Mac Mini M4 (24gb + 16gb), connected via Thunderbolt 4"
    >}}

## Code Migration

The changes to transition your PyTorch code into DDP code can be split into:

1. Set up process groups.
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import mccl

def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("mps:0")

    mccl.apply_thunderbolt_production_defaults(training_defaults=True) # use this when using thunderbolt connection
    dist.init_process_group(backend="mccl", device_id=device)
```

2. Initialise model.
```python
def init_model():
    model = build_model().to(device)
    ddp_model = DDP(model, **kwargs)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
```

3. Cleanup after training.
```python
def cleanup():
    dist.destroy_process_group()
```

The `env` variables `RANK` and `WORLD_SIZE` will be set by `torchrun` when you run:
```bash
# Run on Mac 0 (Master Node):
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=169.254.x.x --master_port=29500 \
    train.py

# Run on Mac 1 (Worker Node):
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=169.254.x.x --master_port=29500 \
    train.py
```

The `master_addr` is the IP address of the Thunderbolt connection on the master node.

## Performance Tuning

All of these are read at **ProcessGroup init**. Defaults come from `mccl/config.py` unless overridden.

#### Transport / networking
| Var | Default | When to override |
| :--- | :--- | :--- |
| MCCL_TRANSPORT | auto | Force tcp or rdma if RDMA is configured |
| MCCL_LISTEN_ADDR | auto | Multi-host: bind address; localhost auto-set for local runs |
| MCCL_PORT_BASE | 29600 | Firewall/port conflicts |
| MCCL_IFNAME | "" | Multi-homed Mac picking wrong interface |
| MCCL_CHUNK_BYTES | 4 MB | Large transfers; TB profile bumps to \ge 16 MB if unset |
| MCCL_SMALL_MSG_THRESHOLD | 256 KiB | Algorithm thresholds for small vs large messages |
| MCCL_CONNECT_TIMEOUT_MS | 30000 | Slow/unreliable links |
| MCCL_SOCK_BUFSIZE | 32 MB (large default) | Set 0 for kernel auto-tune; TB profile uses 32 MB |
| MCCL_TCP_LOWAT | 131072 | macOS TCP throughput tuning |
| MCCL_LINK_PROFILE | unset | Set thunderbolt for TB Mac-to-Mac TCP |
| MCCL_TRANSPORT_CRC | off | Debug corrupted transfers |

#### Compute / GPU sync
| Var | Default | When to override |
| :--- | :--- | :--- |
| MCCL_SYNC_MODE | full (implicit) | Keep full for DDP. Never use coalesced with hook-driven multi-bucket DDP |
| MCCL_EVENT_SYNC | on | Set 0 to disable Metal event path (uses stream sync instead) |
| MCCL_OVERLAP_COMM | on | Overlap comm with GPU; needs event sync |
| MCCL_FAST_MATH | on | Metal kernel precision |
| MCCL_GPU_THRESHOLD | 4096 | When to use GPU vs CPU reduce |
| MCCL_FP32_CPU_REDUCE | off | Set 1 to force CPU vDSP fp32 reduce on UMA |
| MCCL_SHADER_PATH | auto | Dev / non-standard install layout |
| MCCL_ALLREDUCE_ALGO | unset | ring_chunked for large allreduce |
| MCCL_RING_ALGO | unset | chunked / ring_chunked / fast for ring path |

#### Compression
| Var | Default | When to override |
| :--- | :--- | :--- |
| MCCL_COMPRESSION | none | fp16 or topk to cut wire bytes (validate stability) |
| MCCL_TOPK_RATIO | 0.01 | When using topk compression |

#### Runtime / watchdog
| Var | Default | When to override |
| :--- | :--- | :--- |
| MCCL_LOG_LEVEL | WARN | INFO/DEBUG for diagnostics |
| MCCL_WATCHDOG_TIMEOUT_MS | 300000 | Hung collective detection |
| MCCL_HEARTBEAT_INTERVAL_MS | 5000 | Transport keepalive |
| MCCL_MAX_QUEUE_DEPTH | 1024 | Backpressure tuning |

Configure many of these via Python instead:
```python
mccl.init(compression="fp16", chunk_bytes=16*1024*1024)
setup()
```

See more at [docs](https://github.com/mps-ddp/mccl/blob/master/docs/MULTINODE.md).

## Conclusion

I have benchmarked the performance on MCCL on my two-Mac setup and the results are quite remarkable. The details of the benchmark can be found in my github repo

{{< github repo="nveshaan/mccl_benchmark" showThumbnail=false >}}

And, the github repo of MCCL

{{< github repo="mps-ddp/mccl" showThumbnail=false >}}