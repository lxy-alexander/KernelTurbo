import torch
import numpy as np
import random
import logging
from typing import List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    max_error: float
    mean_error: float
    relative_error: float
    validation_details: str = ""


class Validator:
    """通用 CUDA 验证器：只需 PyTorch 代码字符串 + CUDA 文件"""

    def __init__(self,
                 tolerance=1e-3,
                 relative_tolerance=1e-3,
                 device='cuda',
                 seed=42):
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.device = device
        self.seed = seed
        self._set_seed(seed)

    def _set_seed(self, seed: int):
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_input_variants(self,
                                get_inputs: callable,
                                num_variants: int = 10,
                                seed: int = 42) -> List[List[torch.Tensor]]:
        """
        通用输入生成器：根据 get_inputs() 返回的张量模板生成多组测试输入。
        每次运行可复现。
        """
        variants = []
        for i in range(num_variants):
            # 固定每组输入种子，保证可复现
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)
            random.seed(seed + i)

            # 生成输入并复制一份，防止引用问题
            inputs = get_inputs()
            inputs_copy = [x.clone() for x in inputs]
            variants.append(inputs_copy)

        return variants

    def run(self,
            pytorch_module,
            cuda_module,
            get_inputs,
            num_tests=5,
            cuda_func_name="forward"):
        input_generator = get_inputs if get_inputs else (
            lambda: [torch.randn(1, device=self.device)])

        # 验证逻辑（只验证输出一致性）
        results = []  # 存储 ValidationResult 对象
        pytorch_module.eval()
        input_variants = self.generate_input_variants(input_generator,
                                                      num_variants=num_tests)

        for idx, inputs in enumerate(input_variants):
            # 打印每个张量的前几个元素
            previews = []
            for t in inputs:
                if not isinstance(t, torch.Tensor):
                    previews.append(str(t))
                    continue
                if t.ndim == 0:
                    previews.append(str(t.item()))
                elif t.ndim == 1:
                    previews.append(str(t[:5].tolist()))
                else:  # >=2D
                    previews.append(str(t[0, :5].tolist()))
            print(
                f"测试用例 {idx+1}, 输入形状: {[t.shape if isinstance(t, torch.Tensor) else type(t) for t in inputs]}, 前5元素: {previews}"
            )

        for inputs in input_variants:
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = [
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            with torch.no_grad():
                pytorch_out = pytorch_module(*inputs)

            cuda_func = getattr(cuda_module, cuda_func_name)
            cuda_out = cuda_func(*inputs)
            if isinstance(
                    cuda_out,
                    torch.Tensor) and cuda_out.device != pytorch_out.device:
                cuda_out = cuda_out.to(pytorch_out.device)

            diff = (pytorch_out - cuda_out).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            relative_err = (diff /
                            (torch.abs(pytorch_out) + 1e-8)).mean().item()
            
            passed = torch.allclose(pytorch_out, cuda_out, rtol=1e-5, atol=1e-8)

            validation_details = f"max_err={max_err:.2e}, mean_err={mean_err:.2e}, relative_err={relative_err:.2e}"
            
            print(validation_details)
            
            results.append(
                ValidationResult(passed=passed,
                                 max_error=max_err,
                                 mean_error=mean_err,
                                 relative_error=relative_err,
                                 validation_details=validation_details))
        return self.summarize_validation(results)

    def summarize_validation(self,
                             results: List[ValidationResult],
                             test_name="验证"):
        """
        汇总验证结果，返回 (all_passed, summary_str)
        - all_passed: True 表示所有测试通过
        - summary_str: 包含失败案例和平均误差的字符串
        """
        total = len(results)
        passed_count = sum(r.passed for r in results)
        all_passed = passed_count == total

        print(f"\n{'='*60}\n{test_name}\n{'='*60}")
        print(
            f"总测试: {total}, 通过: {passed_count}, 失败: {total - passed_count}, 通过率: {passed_count/total*100:.2f}%"
        )
        for i, r in enumerate(results):
            if not r.passed:
                print(f"失败案例 {i+1}: {r.validation_details}")

        # 收集失败案例信息
        failed_cases = [
            f"案例 {i+1}: {r.validation_details}" for i, r in enumerate(results)
            if not r.passed
        ]
        failed_str = "\n".join(failed_cases) if failed_cases else "无失败案例"

        # 计算平均误差
        avg_max_error = sum(r.max_error for r in results) / total
        avg_mean_error = sum(r.mean_error for r in results) / total
        avg_relative_error = sum(r.relative_error for r in results) / total

        # 拼接字符串
        validate_msg = (
            f"总测试: {total}, 通过: {passed_count}, 失败: {total - passed_count}\n"
            f"失败案例:\n{failed_str}\n"
            f"平均最大误差: {avg_max_error:.6f}\n"
            f"平均平均误差: {avg_mean_error:.6f}\n"
            f"平均相对误差: {avg_relative_error:.6f}")

        return all_passed, validate_msg
