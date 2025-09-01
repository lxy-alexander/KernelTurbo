import textwrap
from utils.status import PipelineStatus
from utils.files import FileUtil


class PromptBuilder:

    @staticmethod
    def create_system_prompt() -> str:
        return "你是CUDA和PyTorch专家，请将给定的PyTorch代码转换为功能等价且高性能的CUDA 实现."

    @staticmethod
    def _task_block() -> str:
        return "任务: CUDA 代码生成与优化"

    @staticmethod
    def _requirements_block() -> str:
        return ("规范要求:\n"
                "- 保持与原PyTorch代码逻辑和功能等价\n"
                "- 去除输入函数和数据\n"
                "- 使用自定义CUDA kernel实现核心计算\n"
                "- 提供主机端接口函数\n"
                "- 模块名必须为 custom_cuda_module 并且只导出一个forward函数\n"
                "- 确保代码可通过torch.utils.cpp_extension.load编译并正确运行\n"
                "- 生成的CUDA 代码请用<cuda></cuda>标签包裹\n"
                "- 你需要参考下面的高性能的代码片段（高性能的代码片段包裹在<fragment></fragment>中间）")

    @staticmethod
    def _high_perf_code_block(high_perf_code: str) -> str:
        return ("高性能代码片段:\n"
                "<fragment>\n"
                f"{high_perf_code}\n"
                "</fragment>")

    @staticmethod
    def _original_code_block(pytorch_code: str) -> str:
        return ("原始代码:\n"
                "<pytorch>\n"
                f"{pytorch_code}\n"
                "</pytorch>")

    @staticmethod
    def _gen_fail_block() -> str:
        return "代码生成错误，请重新按照上述要求重新生成代码。"

    @staticmethod
    def _compile_fail_block(cuda_code: str, compile_error_message: str) -> str:
        return ("待修复代码 (编译错误):\n"
                "<cuda>\n"
                f"{cuda_code}\n"
                "</cuda>\n"
                "编译错误信息:\n"
                f"{compile_error_message or '未知错误'}\n"
                "请在满足规范要求的前提下，认真阅读编译错误信息，并修复代码。")

    @staticmethod
    def _validate_fail_block(cuda_code: str,
                             validation_error_message: str) -> str:
        return ("待修复代码 (验证错误):\n"
                "<cuda>\n"
                f"{cuda_code}\n"
                "</cuda>\n"
                "验证错误信息:\n"
                f"{validation_error_message or '未知错误'}\n"
                "请在满足规范要求的前提下，认真阅读验证错误信息，并修复代码。")

    @staticmethod
    def _success_block(cuda_code: str, performance_metrics: str) -> str:
        return ("待优化代码 (性能优化):\n"
                "<cuda>\n"
                f"{cuda_code}\n"
                "</cuda>\n"
                "性能评估:\n"
                f"{performance_metrics or '无性能数据'}\n"
                "请在规范满足要求的前提下，认真阅读性能评估指标，并继续优化代码使其超越上一次。")

    @staticmethod
    def create_dynamic_prompt(status,
                              pytorch_code,
                              cuda_code=None,
                              compile_error_message=None,
                              validation_error_message=None,
                              performance_metrics=None) -> str:
        high_perf_code_block = """

        """
        
        prompt = [
            PromptBuilder._task_block(),
            PromptBuilder._original_code_block(pytorch_code),
            PromptBuilder._requirements_block(),
            PromptBuilder._high_perf_code_block(high_perf_code_block)
        ]

        if status == PipelineStatus.GEN_FAIL:
            prompt.append(PromptBuilder._gen_fail_block())

        elif status == PipelineStatus.COMPILE_FAIL:
            prompt.append(
                PromptBuilder._compile_fail_block(cuda_code,
                                                  compile_error_message))

        elif status == PipelineStatus.VALIDATE_FAIL:
            prompt.append(
                PromptBuilder._validate_fail_block(cuda_code,
                                                   validation_error_message))

        elif status == PipelineStatus.SUCCESS:
            prompt.append(
                PromptBuilder._success_block(cuda_code, performance_metrics))

        return "\n\n".join(prompt)


if __name__ == "__main__":
    pytorch_code = FileUtil.read_file("raw/pytorch.py")
    cuda_code = pytorch_code
    compile_error_message = "Compile error"
    validation_error_message = None
    perf_metrics = "3x"

    prompt = PromptBuilder.create_dynamic_prompt(
        PipelineStatus.SUCCESS,
        pytorch_code=pytorch_code,
        cuda_code=cuda_code,
        compile_error_message=compile_error_message,
        validation_error_message=validation_error_message,
        performance_metrics=perf_metrics)

    print(prompt)
