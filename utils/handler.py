from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple, Any
from tqdm import tqdm


class Handler:

    def __init__(self,
                 max_workers: int = 1,
                 verbose: bool = True):
        self.max_workers = max_workers
        self.verbose = verbose

    def post(
        self,
        func: Callable,
        arglist: List[Tuple],
    ) -> List[Any]:

        if self.max_workers > 1:
            # Use thread pool for concurrent execution
            return self._run_multi_thread(func, arglist)
        else:
            # Execute sequentially
            return self._run_sequential(func, arglist)

    def _run_sequential(self, func: Callable,
                        arglist: List[Tuple]) -> List[Any]:
        results = []
        for i, args in enumerate(arglist):
            if self.verbose:
                print(f"[Sequential] Task {i + 1}/{len(arglist)}")
            result = func(*args)  # Execute the function with the current args
            results.append(result)
        return results

    def _run_multi_thread(self, func: Callable,
                          arglist: List[Tuple]) -> List[Any]:
        results = []
        self.max_workers = min(self.max_workers, len(arglist))
        # Create a thread pool executor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks to the thread pool
            future_to_args = {
                executor.submit(func, *args): args
                for args in arglist
            }
            # Process results as tasks complete
            for i, future in enumerate(
                    tqdm(as_completed(future_to_args), total=len(arglist))):
                args = future_to_args[future]
                try:
                    result = future.result(
                    )  # Get result of the completed task
                    results.append(result)
                    if self.verbose:
                        print(
                            f"[Thread] ✓ Completed task {i + 1}/{len(arglist)}"
                        )
                except Exception as e:
                    results.append(None)
                    print(
                        f"[Thread Error] ✗ Task {args} failed with exception: {e}"
                    )
        return results
