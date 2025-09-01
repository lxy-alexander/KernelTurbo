import torch
import time
import numpy as np
import random
from typing import List, Callable
from dataclasses import dataclass

# ----------------- 数据类 -----------------
@dataclass
class EvaluatorResult:
    pytorch_time: float
    cuda_time: float
    speedup: float  # CUDA 相比 PyTorch 的加速比

# ----------------- Evaluator -----------------
class Evaluator:
    """
    对 PyTorch 模型与 CUDA 实现进行性能评测（只记录时间和加速比）
    """
    def __init__(self, device='cuda', seed=42):
        self.device = device
        self.seed = seed

    def generate_input_variants(self, get_inputs: Callable, num_variants=5) -> List[List[torch.Tensor]]:
        """根据 get_inputs 模板生成多组不同数据但相同形状的输入"""
        variants = []

        template_inputs = get_inputs()
        shapes = [x.shape for x in template_inputs]
        dtype = [x.dtype for x in template_inputs]
        device = [x.device for x in template_inputs]

        for i in range(num_variants):
            torch.manual_seed(self.seed + i)
            np.random.seed(self.seed + i)
            random.seed(self.seed + i)

            new_inputs = [torch.randn(s, dtype=d, device=dev) for s, d, dev in zip(shapes, dtype, device)]
            variants.append(new_inputs)

        return variants

    def run(self, pytorch_module, cuda_module, get_inputs, num_tests=20, warmup=5, cuda_func_name="forward"):
        results = []
        cuda_func = getattr(cuda_module, cuda_func_name)

        input_variants = self.generate_input_variants(get_inputs, num_variants=num_tests)
        pytorch_module.eval()
        
        for inputs in input_variants:
            inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
            # PyTorch warm-up
            for _ in range(warmup):
                with torch.no_grad():
                    _ = pytorch_module(*inputs)
            # CUDA warm-up
            for _ in range(warmup):
                _ = cuda_func(*inputs)
                
        print(f"[Evaluator]已完成{warmup}次warmup")

        # test
        for _, inputs in enumerate(input_variants):
            inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]

            # PyTorch forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(): # close autograd
                _ = pytorch_module(*inputs)
            torch.cuda.synchronize()
            pytorch_time = time.perf_counter() - t0

            # CUDA forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = cuda_func(*inputs)
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - t0


            speedup = pytorch_time / cuda_time if cuda_time > 0 else float('inf')

            results.append(EvaluatorResult(
                pytorch_time=pytorch_time,
                cuda_time=cuda_time,
                speedup=speedup
            ))

        return pytorch_time, cuda_time, speedup, self.summarize_evaluation(results)
    
    def summarize_evaluation(self, results: List[EvaluatorResult], test_unit="性能测试") -> str:
        total = len(results)
        lines = []
        lines.append(f"{'='*60}\n{test_unit}\n{'='*60}")
        lines.append(f"总测试 = {total}次")
        
        for i, r in enumerate(results):
            test_result = f"测试 {i+1}: PyTorch time={r.pytorch_time:.6f}s, " \
                f"CUDA time={r.cuda_time:.6f}s, 加速比={r.speedup:.2f}x"
            print(test_result)

        avg_pytorch = np.mean([r.pytorch_time for r in results])
        avg_cuda = np.mean([r.cuda_time for r in results])
        avg_speedup = np.mean([r.speedup for r in results])
        
        avg_lines = [
            f"\n平均 PyTorch 时间: {avg_pytorch:.6f}s",
            f"平均 CUDA 时间: {avg_cuda:.6f}s",
            f"平均加速比: {avg_speedup:.2f}x"
        ]
        
        for line in avg_lines:
            print(line)
            lines.append(line)

        return "\n".join(lines)



# ----------------- 示例使用 -----------------

if __name__ == "__main__":
    from compiler import Compiler
    from validator import Validator
    from utils.files import FileUtil as fu
    
    cuda_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3
#define SHARED_SIZE (BLOCK_SIZE + KERNEL_SIZE - 1)
#define MIN_CHANNELS_THRESHOLD 16
#define MIN_SIZE_THRESHOLD 32

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {
    
    __shared__ float shared_input[SHARED_SIZE][SHARED_SIZE];
    __shared__ float shared_weight[KERNEL_SIZE][KERNEL_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;
    const int b = blockIdx.z;
    
    const int x = bx + tx;
    const int y = by + ty;
    
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
                int weight_idx = ((oc * in_channels + ic) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx;
                shared_weight[ty][tx] = weight[weight_idx];
            }
            __syncthreads();
            
            for (int i = ty; i < SHARED_SIZE; i += BLOCK_SIZE) {
                for (int j = tx; j < SHARED_SIZE; j += BLOCK_SIZE) {
                    int ih = by + i - padding;
                    int iw = bx + j - padding;
                    
                    shared_input[i][j] = (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) ?
                        input[((b * in_channels + ic) * input_height + ih) * input_width + iw] : 0.0f;
                }
            }
            __syncthreads();
            
            if (x < output_width && y < output_height) {
                #pragma unroll
                for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
                    #pragma unroll
                    for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                        sum += shared_input[ty * stride + ki][tx * stride + kj] * shared_weight[ki][kj];
                    }
                }
            }
            __syncthreads();
        }
        
        if (x < output_width && y < output_height) {
            output[((b * out_channels + oc) * output_height + y) * output_width + x] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto input_height = x.size(2);
    auto input_width = x.size(3);
    auto out_channels = weight.size(0);
    
    if (input_height < MIN_SIZE_THRESHOLD || input_width < MIN_SIZE_THRESHOLD ||
        in_channels > MIN_CHANNELS_THRESHOLD || out_channels > MIN_CHANNELS_THRESHOLD) {
        return torch::conv2d(x, weight, 
                           bias.has_value() ? bias.value() : torch::Tensor(),
                           {stride, stride}, 
                           {padding, padding},
                           {dilation, dilation},
                           groups);
    }
    
    auto output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
    auto output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width},
                             x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch_size);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        stride,
        padding);
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive CUDA conv2d implementation");
}
    """
    
    pytorch_code = """
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1
groups = 1
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    ]
    """
    
    
    cuda_file=fu.create_temp_cuda_file(cuda_code)
    pytorch_file=fu.create_temp_pytorch_file(pytorch_code)
    
    
    compiler = Compiler(verbose=True)
    

    status, cuda_module, pytorch_module, inputs, log = compiler.run(
        pytorch_file, cuda_file, "deepseek")
    print(status, cuda_module, pytorch_module, inputs, log)

    validator = Validator()
    passed, results = validator.run(pytorch_module, cuda_module, inputs, num_tests=10)
    
    evaluator = Evaluator()
    results = evaluator.run(pytorch_module, cuda_module, inputs)