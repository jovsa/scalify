import os
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    tensorboard_trace_handler,
    record_function,
)

def print_gpu_info():
    print("GPU SPECES:")
    os.system('lspci')

    print("MEMORY SPECS:")
    os.system('nvidia-smi --query-gpu=memory.total --format=csv')

def _run_ops(size=2**13):
    ones = torch.ones((size, size), device=torch.device("cuda:0"))

    torch.matmul(ones, ones, out=ones)      # matrix multiplication
    ones.mul_(0.5)                          # in place multiplication
    result_mul = ones.mul(0.5)              # out of place multiplication
    total = ones + result_mul               # adding tensors
    result_sum = torch.sum(ones)            # adding elements of a tensor
    result_sqrt = torch.sqrt(ones)          # sqrt takes 7 ops
    result_sin = torch.sin(ones)            # sin takes 17 ops (14 fp64, 3 fp32)
    result_sigmoid = torch.sigmoid(ones)    # sigmoid takes 24 ops
    result = torch.log10(ones)              # log10 takes 24 ops
    result = torch.pow(ones, 3.14159)       # pow takes 142 ops


def profile_torch_run(tensorboard=False):

    # warmup run
    _run_ops()

    trace_handler = tensorboard_trace_handler(dir_name="./logs/tb_trace")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=trace_handler if tensorboard else None,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        _run_ops()

    res = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    print(res)

if __name__ == "__main__":
    print_gpu_info()
    profile_torch_run()
