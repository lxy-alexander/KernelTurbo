import os
import tempfile
from pathlib import Path


class FileUtil:
    """Utility class for creating temporary CUDA and PyTorch source files."""

    @staticmethod
    def create_temp_cuda_file(code: str, filename: str = "temp_kernel.cu") -> str:
        temp_dir = tempfile.mkdtemp()
        file_path = Path(temp_dir) / filename
        with open(file_path, "w") as f:
            f.write(code)
        return str(file_path)

    @staticmethod
    def create_temp_pytorch_file(code: str, filename: str = "temp_pytorch.py") -> str:
        temp_dir = tempfile.mkdtemp()
        file_path = Path(temp_dir) / filename
        with open(file_path, "w") as f:
            f.write(code)
        return str(file_path)
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """读取指定路径文件的内容并返回字符串"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return path.read_text()
