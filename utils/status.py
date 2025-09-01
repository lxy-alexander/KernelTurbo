from enum import Enum, auto
from enum import IntEnum

class RequestStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    
class PipelineStatus(IntEnum):
    PENDING = -1      # 未处理
    GEN_FAIL = 0      # 生成失败
    COMPILE_FAIL = 1  # 编译失败
    VALIDATE_FAIL = 2 # 验证失败
    SUCCESS = 3       # 成功