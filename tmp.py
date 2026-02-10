import pandas as pd

df = pd.read_parquet("storage/data/parquets/code_alpaca/train.parquet")

test_func = df['test_func_validated'][0]
print(test_func)
examples = df['train_inputs'][0]
print(examples)

import os
from urllib import response
from typing import Dict, Any
import re
import multiprocessing
import time
from ast import literal_eval
from dataclasses import dataclass
import numpy as np


class RunTestFunc:
    """
    A class to run a test function defined in code.
    """

    def __init__(self, func_code: str, timeout=0.5):
        """
        Initializes the RunTestFunc with the given function code. Is not safe (i.e. runs exec on func_code, ensure you do not run malicious code through here by mistake).

        :param func_code: The code defining the test function. Should come from the provided dataset.
        :type func_code: str
        :param timeout: The maximum time in seconds to allow the function to run.
        :type timeout: float
        """
        self.func_code = func_code
        self.access_counter = 0
        self.attempted_inputs = []
        self.received_outputs = []
        self.timeout = timeout        
        success = self.try_exec(func_code)
        self._context = {"__builtins__": __builtins__}
        if success:
            exec(func_code, self._context)
            self.test_func = self._context["test_func"]
        else:
            raise RuntimeError("Failed to exec function code, cannot initialize RunTestFunc.")


    @staticmethod
    def exec_worker(func_code, queue):
        """Helper worker to run exec and put the result in a queue."""
        try:
            exec(func_code, {"__builtins__": __builtins__})
            queue.put(True) # runs
        except Exception as e:
            queue.put(False) # fails

    def try_exec(self, func_code):
        """Tries to exec the given code in a separate process with a timeout."""
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.exec_worker, args=(func_code, queue))
        p.start()
        p.join(timeout=self.timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return False
        if not queue.empty():
            return queue.get()
        return False

    def worker(self, func, args, queue):
        """Helper worker to run the function and put the result in a queue."""
        try:
            result = func(*args)
            queue.put((result, None))
        except Exception as e:
            queue.put((None, str(e)))

    def run_test(self, *args):
        self.access_counter += 1
        self.attempted_inputs.append(args)

        # Use a Queue to get the return value back from the child process
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=self.worker, args=(self.test_func, args, queue)
        )

        p.start()

        # Wait
        p.join(timeout=self.timeout)

        if p.is_alive():
            # If the process is still running after 15s, kill it
            p.terminate()
            p.join()
            self.received_outputs.append(
                (None, f"Timeout: Function execution exceeded {self.timeout} seconds")
            )
            return None, f"Timeout: Function execution exceeded {self.timeout} seconds"

        # If it finished, grab the result from the queue
        if not queue.empty():
            result = queue.get()
            self.received_outputs.append(result)
            return result
        self.received_outputs.append((None, "Unknown error during execution"))
        return None, "Unknown error during execution"

    def run_test_str(self, args_str: str):
        """
        Runs the test function with the given arguments in string form.

        :param args_str: Arguments in string form to pass to the test function.
        :type args_str: str
        :return: A tuple (return_value, error_message). If there is no error, error_message is None.
        :rtype: tuple
        """
        try:
            args = literal_eval(args_str)  # for safety
        except Exception as e:
            return None, "Invalid input args, is not valid python syntax"
        if not isinstance(args, tuple) and not isinstance(args, list):
            args = (args,)  # for single argument functions
        return self.run_test(*args)
    
runner = RunTestFunc(test_func) 
result = runner.run_test_str(examples[0])
breakpoint()