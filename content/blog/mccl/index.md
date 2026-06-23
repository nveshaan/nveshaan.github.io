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
| Knob | Notes |
|------|--------|
| **`DDP_BUCKET_MB`** | Larger (e.g. 50–200) → **fewer** allreduces per step → fewer TCP round-trips over the inter-host link. Trade-off: memory / peak message size. |
| **`MCCL_COMPRESSION=fp16`** | Halves wire volume when compression is enabled in the build (`ProcessGroupMCCL` compressor path). |
| **FP16 training** | `TRAIN_AUTOCAST_FP16=1` in `ddp_dummy_train.py` (or your script) uses `torch.autocast("mps", dtype=torch.float16)` where supported. |
| **`MCCL_SYNC_MODE=full`** | Required for DDP gradient buckets. **Do not** use `coalesced` with hook-driven DDP (stale grads / broken pipe). |
| **`MCCL_SOCK_BUFSIZE`** | Override kernel socket buffer (bytes); default is large in `Connection.cpp`. Set `0` to let the kernel auto-tune. |
| **`MCCL_CHUNK_BYTES`** | Transport chunk size (see `TransportConfig::from_env()` in `TcpTransport.cpp`); affects CRC/chunked paths. |
| **`MCCL_TRANSPORT`** | `tcp` default; RDMA when available and configured (see `transport/rdma/`). |
| **`MCCL_LINK_PROFILE=thunderbolt`** | Production TCP defaults for Thunderbolt IP links: larger default **socket buffers** and **chunk size**. Use `scripts/thunderbolt_prod.sh` or `mccl.apply_thunderbolt_production_defaults()`. |
| **Model size** | Use larger models (custom `MODEL_HIDDEN`, `MODEL_DEPTH` env vars) to see where DDP becomes worthwhile vs single GPU. |

See more at [docs](https://github.com/mps-ddp/mccl/blob/master/docs/MULTINODE.md).

## Conclusion

I have benchmarked the performance on MCCL on my two-Mac setup and the results are quite remarkable. The details of the benchmark can be found in my github repo

{{< github repo="nveshaan/mccl_benchmark" showThumbnail=false >}}

And, the github repo of MCCL

{{< github repo="mps-ddp/mccl" showThumbnail=false >}}