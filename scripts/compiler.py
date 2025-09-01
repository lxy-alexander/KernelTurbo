import importlib
import os
import subprocess
import sys
import warnings
from typing import Any, Dict, List, Optional

import torch
from torch.utils.cpp_extension import load

class Compiler:
    """CUDA 文件编译器"""

    def __init__(self, verbose: bool = True, device: str = "cuda"):
        self.verbose = verbose
        self.compiled_modules: Dict[str, Any] = {}
        self.device = torch.device(device)
        self._check_environment()

    def _check_environment(self):
        """检查 CUDA 环境和 nvcc 编译器"""
        if not torch.cuda.is_available():
            warnings.warn("CUDA 不可用")
        try:
            result = subprocess.run(['nvcc', '--version'],
                                    capture_output=True,
                                    text=True)
            if result.returncode != 0:
                warnings.warn("nvcc 编译器未找到或不可用")
        except FileNotFoundError:
            warnings.warn("nvcc 编译器未找到")

    def _load_model_components(self, pytorch_path: str):
        module_name = os.path.splitext(os.path.basename(pytorch_path))[0]
        spec = importlib.util.spec_from_file_location(module_name,
                                                      pytorch_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        try:
            model_cls = getattr(module, "Model")
            get_inputs = getattr(module, "get_inputs",
                                 lambda: [torch.randn(1, device=self.device)])
            get_init_inputs = getattr(module, "get_init_inputs", lambda: [])
            return model_cls, get_inputs, get_init_inputs
        except Exception as e:
            raise ImportError(f"加载模块失败: {pytorch_path}\n原因: {e}")
        

    def format_compile_error(e: Exception, max_lines=5, max_chars=200) -> str:
        lines = str(e)
        if not lines:
            return "未知错误"
        if not isinstance(lines, str):
            lines = repr(lines)
        
        lines = lines.splitlines()
        useful = [l for l in lines if "error:" in l.lower() or "failed" in l.lower()]
        if not useful:
            useful = lines  # fallback
        
        # 取前 N 行
        useful = list(useful)  # 确保是 list
        msg = "\n".join(useful[:max_lines])
        
        # 限制最大长度
        return msg[:max_chars]
    
    def run(
        self,
        pytorch_file: str,
        cuda_file: str,
        model_name: str,
        extra_compile_args: Optional[Dict[str, List[str]]] = None,
        extra_link_args: Optional[List[str]] = None,
        with_cuda: bool = True,
    ) -> Any:
        cuda_module_name: str = "custom_cuda_module"
        model_cls, inputs, get_init_inputs = self._load_model_components(
            pytorch_file)
        init_args = get_init_inputs() if get_init_inputs else []
        pytorch_module = model_cls(*init_args).to(self.device)

        extra_compile_args = extra_compile_args or {
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
        }
        extra_link_args = extra_link_args or []

        if self.verbose:
            print(f"[Compiler] 编译模块: {cuda_module_name}")
            print(f"[Compiler] 源文件: {cuda_file}")

        try:
            cuda_module = load(name=cuda_module_name,
                               sources=[cuda_file],
                               extra_cflags=extra_compile_args.get('cxx', []),
                               extra_cuda_cflags=extra_compile_args.get(
                                   'nvcc', []) if with_cuda else [],
                               extra_ldflags=extra_link_args,
                               verbose=self.verbose,
                               with_cuda=with_cuda)
            compile_message = f"编译成功!"
            return True, cuda_module, pytorch_module, inputs, compile_message
        except Exception as e:
            # 捕获异常，不抛出，返回失败标记和日志
            compile_message = f"编译失败: {self.format_compile_error(e)}"
            return False, None, pytorch_module, inputs, compile_message

    def get_module(self, module_name: str) -> Optional[Any]:
        """获取已编译模块"""
        return self.compiled_modules.get(module_name)

    def list_modules(self) -> List[str]:
        """列出所有已编译模块"""
        return list(self.compiled_modules.keys())

    def clear_cache(self):
        """清空已编译模块缓存"""
        self.compiled_modules.clear()

