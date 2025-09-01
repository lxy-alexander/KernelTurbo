import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import torch
import gc

from db.db_manager import DBManager
from llm.llm_manager import ModelManager
from prompt.prompt_builder import PromptBuilder as pb
from utils.status import PipelineStatus
from utils.files import FileUtil as fu
from utils.handler import Handler

from scripts.generator import Generator
from scripts.compiler import Compiler
from scripts.validator import Validator
from scripts.evaluator import Evaluator


load_dotenv()

# ---------------- Pipeline ----------------
class Pipeline:

    def __init__(self, db_name, pytorch_path, max_workers = 8, max_rounds = 5):
        self.max_workers = max_workers
        self.status = None
        self.pytorch_code = fu.read_file(pytorch_path)
        self.cuda_code = None
        self.max_rounds = max_rounds
        self.gpu_lock = threading.Lock()

        self.system_prompt = pb.create_system_prompt()
        self.prompt = pb.create_dynamic_prompt(self.status, self.pytorch_code)

        self.generator = Generator(self.system_prompt)
        self.compiler = Compiler()
        self.validator = Validator()
        self.evaluator = Evaluator()

        self.db_manager = DBManager(db_name= db_name)

        self.experiment_id = self.db_manager.insert_experiment("first experiment")
        self.prompt_id = self.db_manager.insert_prompt(self.system_prompt, self.prompt, "性能优化")

    def model_worker(self, llm):
        print(f"[PID {os.getpid()}] Starting model: {llm.model}")
        pytorch_file = fu.create_temp_pytorch_file(self.pytorch_code)
        model_id = self.db_manager.insert_model(llm.model, llm.provider)
        
        iter_prompt = self.prompt
        status = self.status
        
        for iter in range(1, self.max_rounds + 1):
            status = PipelineStatus.PENDING
            compile_msg, validate_msg = None, None
            pytorch_code_exec_time, cuda_code_exec_time, speedup, perf_metrics = None, None, None, None

            try:
                # 生成
                gen_status, cuda_code = self.generator.run(llm, iter_prompt)
                cuda_file = fu.create_temp_cuda_file(cuda_code)
                if not gen_status:
                    status = PipelineStatus.GEN_FAIL
                    continue

                # 编译
                compile_status, cuda_module, pytorch_module, inputs, compile_msg = self.compiler.run(
                    pytorch_file, cuda_file, llm.model
                )
                if not compile_status:
                    status = PipelineStatus.COMPILE_FAIL
                    continue

                # 验证 + 性能评估
                validate_status, validate_msg = self.validator.run(
                    pytorch_module, cuda_module, inputs, num_tests=5
                )
                if not validate_status:
                    status = PipelineStatus.VALIDATE_FAIL
                    continue

                pytorch_code_exec_time, cuda_code_exec_time, speedup, perf_metrics = self.evaluator.run(
                    pytorch_module, cuda_module, inputs
                )
                status = PipelineStatus.SUCCESS

            finally:
                self.cleanup_gpu_memory()
                iter_prompt = pb.create_dynamic_prompt(status, self.pytorch_code, cuda_code, compile_msg, validate_msg, perf_metrics)
                gen_id = self.db_manager.insert_generation(
                    iter, self.pytorch_code, cuda_code, self.prompt,
                    model_id, self.prompt_id, self.experiment_id
                )
                self.db_manager.insert_benchmark(
                    gen_id, compile_msg, validate_msg,
                    pytorch_code_exec_time, cuda_code_exec_time, speedup,
                    self.status
                )

        print(f"[PID {os.getpid()}] Finished model: {llm.model}")
        
    def cleanup_gpu_memory(self):
        """清理GPU内存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"[WARNING] GPU cleanup failed: {e}")

    def run(self):
        llms = ModelManager().load_models_from_config()
        for llm in llms:
            self.model_worker(llm)
        # max_workers = min(len(llms), self.max_workers)
        # hander = Handler(max_workers)
        # tasks = [(llm,) for llm in llms]
        # hander.post(func = self.model_worker, arglist = tasks)
        self.db_manager.shutdown()
        print("Pipeline finished and results saved to database.")


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    
    #要加如果cuda不可用直接退出
    pytorch_path = "raw/pytorch.py"
    pipeline = Pipeline("kernel_turbo_deepseek_chat_v1.db", pytorch_path, max_rounds=10)
    pipeline.run()
