---
title: "Distributed Training on Apple Silicon via MCCL"
date: 2026-06-20
draft: false
tags: ["deep learning", "pytorch", "mac"]
---

## Overview

**Collective Communication Libraries** such as `nccl` for nvidia gpus, `rccl` for amd gpus and `xccl` for intel gpus have enabled to scale up training of deep learning models to multiple gpu clusters. For apple gpus however, there exists `jaccl` but is exclusively for the **MLX Framework**. There was no way to use **PyTorch DDP** on Apple Silicon, up until the developers at [mps-ddp](https://github.com/mps-ddp) had created and graciously open-sourced [`mccl`](https://github.com/mps-ddp/mccl). The rest of the blog is a guide to do distributed training with PyTorch DDP on MCCL.

{{< figure
    src="mccl.png"
    alt="mac setup"
    caption="2 Mac Mini M4 (24gb + 16gb)"
    >}}

## Code

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import mccl

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("mps:0")

    dist.init_process_group(backend="mccl", device_id=device)

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    ddp_model = DDP(model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(10):
        x = torch.randn(8, 512, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss_fn(ddp_model(x), y).backward()
        optimizer.step()
        if rank == 0:
            print(step, "ok")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```