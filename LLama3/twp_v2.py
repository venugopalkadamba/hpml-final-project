import os
from typing import Optional
from dataclasses import dataclass, field
import time
from tqdm import tqdm
import functools
import numpy as np

# import fastckpt before transformers
from lightseq_ckpt_monkey_patch import replace_hf_ckpt_with_new_ckpt, clear_all_buffers_at_the_end_of_training
replace_hf_ckpt_with_new_ckpt()

import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name    
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import torch.profiler as profiler

from async_communication import reset_global_memory_buffer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    rope_scaling_mode: Optional[str] = field(default=None)
    rope_scaling_factor: Optional[int] = field(default=1.0)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def calculate_mha_flops(batch_size, seq_len, hidden_size, num_heads):
    """Calculate FLOPs for multi-head attention"""
    # Q, K, V projections
    projection_flops = 3 * batch_size * seq_len * hidden_size * hidden_size
    # Attention scores
    attention_flops = batch_size * num_heads * seq_len * seq_len * (hidden_size // num_heads)
    # Attention weighted sum
    weighted_sum_flops = batch_size * num_heads * seq_len * seq_len * (hidden_size // num_heads)
    # Output projection
    output_flops = batch_size * seq_len * hidden_size * hidden_size
    
    return projection_flops + attention_flops + weighted_sum_flops + output_flops

def calculate_mlp_flops(batch_size, seq_len, hidden_size, intermediate_size):
    """Calculate FLOPs for MLP layers"""
    # First linear layer
    first_linear_flops = batch_size * seq_len * hidden_size * intermediate_size
    # Activation (assuming GELU)
    activation_flops = batch_size * seq_len * intermediate_size
    # Second linear layer
    second_linear_flops = batch_size * seq_len * intermediate_size * hidden_size
    
    return first_linear_flops + activation_flops + second_linear_flops

class FLOPsProfiler:
    def __init__(self, model):
        self.model = model
        self.total_flops = 0
        self.total_memory = 0
        self.activation_memory = 0
        self.parameter_memory = 0

    def profile_forward_pass(self, batch_size, seq_length):
        config = self.model.config
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        # Calculate FLOPs for each transformer layer
        per_layer_flops = (
            calculate_mha_flops(batch_size, seq_length, hidden_size, num_heads) +
            calculate_mlp_flops(batch_size, seq_length, hidden_size, intermediate_size)
        )
        
        total_flops = per_layer_flops * num_layers
        
        # Calculate memory usage
        dtype_size = 2  # fp16 = 2 bytes
        
        # Activation memory (estimated)
        self.activation_memory = (
            batch_size * seq_length * hidden_size * num_layers * 4 * dtype_size
        )
        
        # Parameter memory
        total_params = sum(p.numel() for p in self.model.parameters())
        self.parameter_memory = total_params * dtype_size
        
        self.total_flops = total_flops
        self.total_memory = self.activation_memory + self.parameter_memory
        
        return {
            'flops': self.total_flops,
            'memory': self.total_memory,
            'activation_memory': self.activation_memory,
            'parameter_memory': self.parameter_memory
        }

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_config(config).to(device="cuda", dtype=torch.float16)
    model.model.gradient_checkpointing = True

    def initialize_distributed():
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print("torch distributed is already initialized, "
                      "skipping initialization ...", flush=True)
        else:
            if int(os.environ["RANK"]) == 0:
                print("Initializing Torch distributed.")
            dist.init_process_group(backend="nccl")
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            torch.cuda.set_device(dist.get_rank() % local_world_size)

    initialize_distributed()
    transformer_cls_to_wrap = set()
    transformer_cls = get_module_class_from_name(model, "LlamaDecoderLayer")
    if transformer_cls is None:
        raise Exception("Could not find the transformer layer class to wrap in the model.")
    else:
        transformer_cls_to_wrap.add(transformer_cls)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    rank0_print(f"trainable params {params}")

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01
    )

    # Initialize the FLOPs profiler
    flops_profiler = FLOPsProfiler(model)

    for length in [512]:
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA
            ],
            on_trace_ready=profiler.tensorboard_trace_handler('./log'),
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
        ) as prof:
            time_list = []
            data = {
                "input_ids": torch.randint(0, 1000, (1, length,), device="cuda"),
                "labels": torch.randint(0, 1000, (1, length,), device="cuda"), 
                "attention_mask": torch.ones(1, length, device="cuda")
            }
            
            # Profile theoretical FLOPs
            metrics = flops_profiler.profile_forward_pass(
                batch_size=data["input_ids"].shape[0],
                seq_length=data["input_ids"].shape[1]
            )
            
            for i in tqdm(range(10)):
                if i > 0:
                    time_s = time.time()
                with record_function("forward_pass"):
                    outputs = model(**data, use_cache=True)
                with record_function("backward_pass"):
                    outputs.loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()
                if i > 0:
                    time_e = time.time()
                    time_list.append(time_e - time_s)
                    print(f"Cumulated time list {time_list}")
            
            time_arr = np.asarray(time_list) * 1000.0
            
        # Get memory metrics from PyTorch profiler
        memory_stats = prof.key_averages()
        total_cuda_memory = sum(event.cuda_memory_usage for event in memory_stats)
        
        arithmetic_intensity = metrics['flops'] / metrics['memory'] if metrics['memory'] > 0 else 0
        
        print(f"Per GPU shape: {data['input_ids'].shape}")
        print(f"Mean Latency: {np.mean(time_arr):.2f} ms")
        print(f"Std Latency: {np.std(time_arr):.2f} ms")
        print(f"Theoretical FLOPs: {metrics['flops']:,}")
        print(f"Total Memory: {metrics['memory']:,} bytes")
        print(f"Activation Memory: {metrics['activation_memory']:,} bytes")
        print(f"Parameter Memory: {metrics['parameter_memory']:,} bytes")
        print(f"Arithmetic Intensity: {arithmetic_intensity:.2f}")
        print(f"Measured CUDA Memory: {total_cuda_memory:,} bytes")
        
        reset_global_memory_buffer()
        clear_all_buffers_at_the_end_of_training()

if __name__ == "__main__":
    train()