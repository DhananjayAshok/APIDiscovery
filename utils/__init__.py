# These are all the utils functions or classes that you may want to import in your project
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error, log_info, log_warn, log_dict
from utils.hash_handling import write_meta, add_meta_details
from utils.plot_handling import Plotter, sns, plt
from utils.fundamental import file_makedir
from utils.lm_inference import HuggingFaceModel, OpenAIModel, call_infer
import multiprocessing
from ast import literal_eval


def get_test_func_header(func_code):
    """
    Extracts the first test_func header
    """
    header_start = func_code.index("def test_func(")
    header_end = func_code.index("\n", header_start)
    func_header = func_code[header_start:header_end]
    return func_header.strip()


def get_header(func_code: str) -> str:
    """
    Extracts the first function header
    """
    header_start = func_code.index("def ")
    header_end = func_code.index("\n", header_start)
    func_header = func_code[header_start:header_end]
    return func_header.strip()


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
            raise RuntimeError(
                "Failed to exec function code, cannot initialize RunTestFunc."
            )

    @staticmethod
    def exec_worker(func_code, queue):
        """Helper worker to run exec and put the result in a queue."""
        try:
            exec(func_code, {"__builtins__": __builtins__})
            queue.put(True)  # runs
        except Exception as e:
            queue.put(False)  # fails

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


def get_initial_results(func_code, examples):
    try:
        runner = RunTestFunc(func_code)
    except:
        return None, None
    prev_results = []
    example_outputs = []
    for example in examples:
        example_outputs.append(runner.run_test_str(example))
    for i, example_input in enumerate(examples):
        input_str = example_input
        output, err = example_outputs[i]
        prev_results.append((input_str, output, err))
    return prev_results, runner


def get_prev_results_str(prev_results, max_previous_results):
    if not prev_results:
        return "[]"
    results_str = "[\n"
    to_slice = (
        prev_results[-max_previous_results:]
        if max_previous_results is not None
        else prev_results
    )
    for inp, out, err in to_slice:
        results_str += f"  Input: {inp} => Output: {out}, Error: {err}\n"
    results_str += "]"
    return results_str


def get_in_context_prompt(func_header, previous_examples):
    pass


first_reasoning_prompt = f"""
You are given a Python function with the following header:
[HEADER]
Your task is to try various inputs to discover what this function does.

So far, you have tried the following inputs: [PREV]
You then came up with the following running hypothesis: [HYPOTHESIS]

Based on this, what kind of input will you use to test the function with next? Very briefly describe your next intended input only, and the properties it satisfies. How does this input help test the hypothesis? What is the expected output? Be extremely concise and short. 
Your response should be extremely short and concise, just a few sentences. After the response, say [STOP]
Now provide your reasoning below and then say [STOP]
Reasoning:"""


def get_zero_shot_starting_prompt(func_header, previous_examples, full_fill=True):
    prev_str = get_prev_results_str(previous_examples, max_previous_results=None)
    prompt = first_reasoning_prompt.replace("[HEADER]", func_header)
    if full_fill:
        prompt = prompt.replace("[PREV]", prev_str)
        prompt = prompt.replace("[HYPOTHESIS]", "Not yet formed")
    return prompt


def get_zeroshot_starting_details(func_code, examples):
    prev_results, runner = get_initial_results(func_code, examples)
    func_header = get_test_func_header(func_code)
    prompt = get_zero_shot_starting_prompt(func_header, prev_results, full_fill=False)
    return prompt, runner, prev_results, func_header
