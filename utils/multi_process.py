from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple, Any
from tqdm import tqdm


class MultiProcess:
    def __init__(self, multi_process=True, max_workers=8, verbose=True):
        self.multi_process = multi_process
        self.max_workers = max_workers
        self.verbose = verbose

    def post(self, func: Callable, arglist: List[Tuple]) -> List[Any]:
        if self.multi_process:
            return self._run_multi_process(func, arglist)
        else:
            return self._run_sequential(func, arglist)

    def _run_sequential(self, func: Callable, arglist: List[Tuple]) -> List[Any]:
        results = []
        for i, args in enumerate(arglist):
            if self.verbose:
                print(f"[Sequential] Task {i + 1}/{len(arglist)}")
            results.append(func(*args))
        return results

    def _run_multi_process(self, func: Callable, arglist: List[Tuple]) -> List[Any]:
        results = []
        self.max_workers = min(self.max_workers, len(arglist))
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_args = {executor.submit(func, *args): args for args in arglist}
            for i, future in enumerate(tqdm(as_completed(future_to_args), total=len(arglist))):
                args = future_to_args[future]
                try:
                    result = future.result()
                    results.append(result)
                    if self.verbose:
                        print(f"[Process] ✓ Completed task {i + 1}/{len(arglist)}")
                except Exception as e:
                    results.append(None)
                    print(f"[Process Error] ✗ Task {args} failed with exception: {e}")
        return results


import time

def sample_task(x, y):
    """简单加法任务，模拟延迟"""
    time.sleep(1)
    return x + y

def task_with_error(x):
    """当x为负数时抛异常"""
    time.sleep(0.1)
    if x < 0:
        raise ValueError(f"Negative value: {x}")
    return x * 2

if __name__ == "__main__":
    mp = MultiProcess(multi_process=False, verbose=True)
    args = [(i, i*2) for i in range(5)]
    print("=== 顺序执行测试 ===")
    results = mp.post(sample_task, args)
    print("结果:", results)
    assert results == [0, 3, 6, 9, 12]

    mp = MultiProcess(multi_process=True, max_workers=3, verbose=True)
    print("\n=== 多进程执行测试 ===")
    results = mp.post(sample_task, args)
    print("结果:", results)
    assert sorted(results) == [0, 3, 6, 9, 12]  # 无序返回结果时用sorted校验

    # print("\n=== 异常处理测试 ===")
    # mp = MultiProcess(multi_process=True, max_workers=2, verbose=True)
    # args_with_error = [(i,) for i in [1, -1, 2, -2, 3]]
    # results = mp.post(task_with_error, args_with_error)
    # print("结果:", results)
    # # 负数输入应返回None（异常捕获时）
    # expected = [2, None, 4, None, 6]
    # assert results == expected