# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Profiling utilities for training loop."""

from typing import Optional

import torch
import torch.profiler

from megatron.bridge.training.config import ProfilingConfig
from megatron.bridge.utils.common_utils import print_rank_0

# Type alias for NVTX context manager
TNvtxContext = torch.autograd.profiler.emit_nvtx


def should_profile_rank(config: Optional[ProfilingConfig], rank: int) -> bool:
    """Check if current rank should be profiled.

    Args:
        config: Profiling configuration
        rank: Current process rank

    Returns:
        True if this rank should be profiled
    """
    if config is None:
        return False
    return rank in config.profile_ranks


def handle_profiling_step(
    config: Optional[ProfilingConfig],
    iteration: int,
    rank: int,
    pytorch_prof: Optional[torch.profiler.profile],
) -> Optional[TNvtxContext]:
    """Handle profiling logic for a single training step.

    Args:
        config: Profiling configuration
        iteration: Current training iteration
        rank: Current process rank
        pytorch_prof: PyTorch profiler instance (if using PyTorch profiler)

    Returns:
        NVTX context if nsys profiling was started at this step, None otherwise
    """
    if not should_profile_rank(config, rank):
        return None

    if config.use_pytorch_profiler and pytorch_prof is not None:
        pytorch_prof.step()

    if config.use_nsys_profiler:
        if iteration == config.profile_step_start:
            return start_nsys_profiler(config)

    return None


def handle_profiling_stop(
    config: Optional[ProfilingConfig],
    iteration: int,
    rank: int,
    pytorch_prof: Optional[torch.profiler.profile],
    nsys_nvtx_context: Optional[TNvtxContext] = None,
) -> None:
    """Handle profiling cleanup at designated stop iteration.

    Args:
        config: Profiling configuration
        iteration: Current training iteration
        rank: Current process rank
        pytorch_prof: PyTorch profiler instance (if using PyTorch profiler)
        nsys_nvtx_context: NVTX context from handle_profiling_step (if using nsys profiler)
    """
    if not should_profile_rank(config, rank):
        return

    if iteration != config.profile_step_end:
        return

    if config.use_pytorch_profiler and pytorch_prof is not None:
        pytorch_prof.stop()

    if config.use_nsys_profiler:
        stop_nsys_profiler(nsys_nvtx_context)


def initialize_pytorch_profiler(
    config: ProfilingConfig,
    tensorboard_dir: str,
) -> torch.profiler.profile:
    """Initialize PyTorch profiler with config settings.

    Args:
        config: Profiling configuration
        tensorboard_dir: Directory for tensorboard outputs

    Returns:
        Initialized (but not started) PyTorch profiler
    """
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(config.profile_step_start - 1, 0),
            warmup=1 if config.profile_step_start > 0 else 0,
            active=config.profile_step_end - config.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir),
        record_shapes=config.record_shapes,
        with_stack=True,
    )
    return prof


def start_nsys_profiler(config: ProfilingConfig) -> TNvtxContext:
    """Start CUDA profiler for nsys profiling.

    Args:
        config: Profiling configuration

    Returns:
        NVTX context manager that must be passed to stop_nsys_profiler
    """
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
    if config.record_shapes:
        nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=True)
    else:
        nvtx_context = torch.autograd.profiler.emit_nvtx()
    nvtx_context.__enter__()
    return nvtx_context


def stop_nsys_profiler(nvtx_context: Optional[TNvtxContext]) -> None:
    """Stop CUDA profiler for nsys profiling.

    Args:
        nvtx_context: NVTX context manager returned from start_nsys_profiler
    """
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
    if nvtx_context is not None:
        nvtx_context.__exit__(None, None, None)

def _print_gpu_memory_usage(stage_name: str) -> None:
    """Print GPU memory usage at a specific stage.
    
    Args:
        stage_name: Name of the stage for logging purposes
    """
    if not torch.cuda.is_available():
        print_rank_0(f"[{stage_name}] CUDA not available, skipping memory report")
        return
    
    # Get memory stats
    reserved = torch.cuda.memory_reserved() / 1e9  # GB
    
    # Get device properties for total memory
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # GB
    
    print_rank_0(f"[{stage_name}] GPU Memory Usage:")
    print_rank_0(f"  Reserved:  {reserved:.2f} GB")
    print_rank_0(f"  Free: {(total_memory - reserved):.2f} GB")

def get_tensor_dict_size(d):
    """Recursively calculate actual tensor memory in a nested dict."""
    total = 0
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            total += value.element_size() * value.numel()
        elif isinstance(value, dict):
            total += get_tensor_dict_size(value)
        elif hasattr(value, 'data') and isinstance(value.data, torch.Tensor):
            # Handle ShardedTensor or similar wrapper objects
            total += value.data.element_size() * value.data.numel()
        elif hasattr(value, 'local_shards'):
            # Handle ShardedTensor with local_shards attribute
            for shard in value.local_shards:
                if hasattr(shard, 'tensor') and isinstance(shard.tensor, torch.Tensor):
                    total += shard.tensor.element_size() * shard.tensor.numel()
        elif hasattr(value, '_tensor') and isinstance(value._tensor, torch.Tensor):
            # Handle objects with _tensor attribute
            total += value._tensor.element_size() * value._tensor.numel()
    return total

def get_tensor_size(value):
    """Get size of a single tensor or wrapped tensor object."""
    if isinstance(value, torch.Tensor):
        return value.element_size() * value.numel()
    elif hasattr(value, 'data') and isinstance(value.data, torch.Tensor):
        return value.data.element_size() * value.data.numel()
    elif hasattr(value, 'local_shards'):
        total = 0
        for shard in value.local_shards:
            if hasattr(shard, 'tensor') and isinstance(shard.tensor, torch.Tensor):
                total += shard.tensor.element_size() * shard.tensor.numel()
        return total
    elif hasattr(value, '_tensor') and isinstance(value._tensor, torch.Tensor):
        return value._tensor.element_size() * value._tensor.numel()
    return 0

def print_itemized_state_dict_size(d, prefix="", max_depth=5, current_depth=0):
    """Recursively print itemized sizes of state dict components."""
    if current_depth >= max_depth:
        return
    
    items = []
    for key, value in d.items():
        if isinstance(value, dict):
            size = get_tensor_dict_size(value)
            items.append((key, value, size, True))
        else:
            size = get_tensor_size(value)
            items.append((key, value, size, False))
    
    # Sort by size (largest first)
    items.sort(key=lambda x: x[2], reverse=True)
    
    for key, value, size, is_dict in items:
        if size == 0:
            continue  # Skip zero-size items
        
        # Format size
        if size >= 1024**3:
            size_str = f"{size / (1024**3):.2f} GB"
        elif size >= 1024**2:
            size_str = f"{size / (1024**2):.2f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} B"
        
        # Print with indentation
        indent = "  " * current_depth
        print_rank_0(f"{indent}{prefix}{key}: {size_str}")
        
        # Recurse into dictionaries
        if is_dict and current_depth < max_depth - 1:
            print_itemized_state_dict_size(value, prefix="", max_depth=max_depth, current_depth=current_depth + 1)
