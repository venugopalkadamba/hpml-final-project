import os
from typing import Optional
from dataclasses import dataclass, field
import time
from tqdm import tqdm
import functools
import numpy as np
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name    
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

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


def compute_arithmetic_intensity(flops, memory_bytes):
    """Compute arithmetic intensity: FLOPs per byte."""
    return flops / memory_bytes if memory_bytes > 0 else 0


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
            global_world_size = dist.get_world_size()
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

    for length in [512]:  # Test for different sequence lengths
        time_list = []
        data = {
            "input_ids": torch.randint(0, 1000, (1, length), device="cuda"),
            "labels": torch.randint(0, 1000, (1, length), device="cuda"),
            "attention_mask": torch.ones(1, length, device="cuda"),
        }

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            for i in tqdm(range(10)):
                if i > 0:
                    time_s = time.time()
                with record_function("model_forward_backward"):
                    outputs = model(**data, use_cache=True)
                    outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i > 0:
                    time_e = time.time()
                    time_list.append(time_e - time_s)

        time_arr = np.asarray(time_list) * 1000.0
        rank0_print(f"Per GPU shape: {data['input_ids'].shape}, "
                    f"Mean Latency: {np.mean(time_arr):.2f} ms, "
                    f"Std Latency: {np.std(time_arr):.2f} ms")

        flops = prof.key_averages().total_average().flops
        memory_bytes = prof.key_averages().total_average().cuda_memory_usage
        arithmetic_intensity = compute_arithmetic_intensity(flops, memory_bytes)

        rank0_print(f"FLOPs: {flops}, Memory: {memory_bytes} bytes, "
                    f"Arithmetic Intensity: {arithmetic_intensity:.2f}")

if __name__ == "__main__":
    train()