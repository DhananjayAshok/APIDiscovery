import os
from urllib import response
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
import re
import multiprocessing
import time
from ast import literal_eval
from openai import OpenAI
from dataclasses import dataclass


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
        locals = {}
        global_dict = globals().copy()
        if success:
            exec(func_code, global_dict, locals)
        else:
            raise RuntimeError(
                "Failed to exec function code, cannot initialize RunTestFunc."
            )
        self.test_func = locals["test_func"]

    @staticmethod
    def exec_worker(func_code, queue):
        """Helper worker to run exec and put the result in a queue."""
        try:
            exec(func_code, globals())
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

    @staticmethod
    def worker(func, args, queue):
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


@dataclass
class LLMJudgeEnvConfig:
    model: str = "gpt-4o-mini"
    base_url = None


class FunctionDiscoveryEnv(BaseTextEnv):
    """
    Environment for multiplication.
    """

    reasoning_prompt = f"""
    You are given a Python function with the following header:
    [HEADER]
    Your task is to try various inputs to discover what this function does.

    So far, you have tried the following inputs: [PREV]
    You then came up with the following running hypothesis: [HYPOTHESIS]

    Based on this, what kind of input will you use to test the function with next? Very briefly describe your next intended input only, and the properties it satisfies. How does this input help test the hypothesis? What is the expected output? Be extremely concise and short. 
    Your response should be extremely short and concise, just a few sentences. After the response, say [STOP]
    Now provide your reasoning below and then say [STOP]
    Reasoning:"""

    input_prompt = f"""
    You are given a Python function with the following header:
    [HEADER]
    Your task is to try various inputs to discover what this function does.

    So far, you have tried the following inputs: [PREV]
    You then came up with the following running hypothesis: [HYPOTHESIS]

    Based on this, you wanted to try the following kind of input next: [REASONING]. 
    Now, give the exact input to test the function with next.
    The input should be valid Python tuples and your output should follow the format below.
    Suggested Input:
    (arg0, arg1) [STOP] #(arg0, arg1) should be replaced with actual input values in your response and must be a valid python tuple. This is an example format for a two arg function. You should adjust the number of arguments as per the function definition.
    Now provide your suggested inputs below and then say [STOP]
    Suggested Input:"""

    reflection_prompt = f"""
    You are given a Python function with the following header:
    [HEADER]
    Your task is to try various inputs to discover what this function does.

    So far, you have tried the following inputs: [PREV]
    You then came up with the following running hypothesis: [HYPOTHESIS]
    You wanted to test this, with an input coming from the reasoning: [REASONING]
    Finally, you just tried the following inputs: [LAST_INPUTS]

    Based on this, can you conclude with very high confidence what the function does? If the function did not perform as you expected, the answer is likely no. If you think it is yes, then say YES and provide a concise description of its functionality.
    Else, say NO and provide a revised hypothesis of what you think the function may do, and some guidance on how to test this further.
    Format Example:
    Hypothesis Conclusion: YES/NO
    Summary: <your extremely concise summary or brief revised hypothesis here>
    [STOP]

    Now, provide your conclusion below, remember to say [STOP] after your summary.
    Hypothesis Conclusion:"""

    judge_hypothesis_prompt = f"""
    You are given the following Python function:
    [FUNCTION]

    What the function truly does is: [FUNCTIONALITY]
    A student is trying to understand what this function does by testing it with various inputs. They have come up with the following hypothesis about the function's behavior:
    [HYPOTHESIS]
    How good is this hypothesis? Please rate it on a scale of 0 to 9, where 0 means the hypothesis is completely incorrect and 9 means the hypothesis is completely correct. Please provide a brief explanation for your rating.
    Answer format:
    Explanation: <your brief explanation here>
    Rating: <a number from 0 to 9>
    [STOP]
    Now, provide your evaluation below, remember to follow the answer format and say [STOP] at the end.
    Evaluation:
    """

    judge_reasoning_prompt = f"""
    You are given the following Python function:
    [FUNCTION]

    What the function truly does is: [FUNCTIONALITY]
    A student is trying to understand what this function does by testing it with various inputs. They have come up with the following hypothesis about the function's behavior:
    [HYPOTHESIS]
    Their reasoning for the next input they want to test is as follows:
    [REASONING]
    How good is this reasoning for testing the hypothesis? Please rate it on a scale of 0 to 9, where 0 means the reasoning is completely ineffective for testing the hypothesis and 9 means the reasoning is extremely effective for testing the hypothesis. Please provide a brief explanation for your rating.
    Answer format:
    Explanation: <your brief explanation here>
    Rating: <a number from 0 to 9>
    [STOP]
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "test_func_validated" in extras, "test_func_validated field is required"
        assert "description" in extras, "description field is required"
        assert "train_inputs" in extras, "train_inputs field is required"
        assert "test_inputs" in extras, "test_inputs field is required"
        self.test_func_validated = extras["test_func_validated"]
        self.description = extras["description"]
        self.train_inputs = extras["train_inputs"]
        self.test_inputs = extras["test_inputs"]
        self.max_turns = 40
        self.max_previous_results = 5
        try:
            self.runner = RunTestFunc(self.test_func_validated)
        except:
            self.runner = None
        if self.runner is not None:
            func_code = self.test_func_validated
            header_start = func_code.index("def test_func(")
            header_end = func_code.index("\n", header_start)
            func_header = func_code[header_start:header_end]
            self.func_header = func_header
            self.reasoning_prompt_filled = self.reasoning_prompt.replace(
                "[HEADER]", self.func_header
            )
            self.input_prompt_filled = self.input_prompt.replace(
                "[HEADER]", self.func_header
            )
            self.reflection_prompt_filled = self.reflection_prompt.replace(
                "[HEADER]", self.func_header
            )
            self.prev_results = []
            example_outputs = []
            for example in self.train_inputs:
                example_outputs.append(self.runner.run_test_str(example))

            for i, example_input in enumerate(self.train_inputs):
                input_str = example_input
                output, err = example_outputs[i]
                self.prev_results.append((input_str, output, err))
            self.concluded = False
            self.turn_kind = "input"
            self.current_hypothesis = "First Turn. No hypothesis yet."
            self.previous_reasoning = None
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError("`OPENAI_API_KEY` must be set for Llm as a judge env")
            self.llm_judge_client = OpenAI(
                base_url=env_config.base_url, api_key=openai_api_key
            )
            self.model = env_config.model

    def judge_infer(self, prompt, max_new_tokens=100):
        response = self.llm_judge_client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_new_tokens,
        )
        text = response.output_text
        if "[STOP]" in text:
            text = text.split("[STOP]")[0]
        return text.strip()

    def get_prev_results_str(self):
        if not self.prev_results:
            return "[]"
        results_str = "[\n"
        to_slice = (
            self.prev_results[-self.max_previous_results :]
            if self.max_previous_results is not None
            else self.prev_results
        )
        for inp, out, err in to_slice:
            results_str += f"  Input: {inp} => Output: {out}, Error: {err}\n"
        results_str += "]"
        return results_str

    def get_hypothesis_reward(self, decision: str, clean_parse: bool):
        hypothesis_prompt_filled = self.judge_hypothesis_prompt.replace(
            "[FUNCTION]",
            self.test_func_validated,
            "[FUNCTIONALITY]",
            self.description,
            "[HYPOTHESIS]",
            self.current_hypothesis,
        )
        evaluation = self.judge_infer(hypothesis_prompt_filled)
        # find rating: digit
        rating_match = re.search(r"rating:\s*(\d+)", evaluation.lower())
        hypothesis_score = 0.0
        if rating_match:
            rating = int(rating_match.group(1))
            hypothesis_score = rating / 9.0  # normalize to [0, 1]
        else:
            return 0.0
        termination_score = 0
        if decision.lower() == "yes":
            if hypothesis_score > 0.7:
                termination_score = 1.0
            else:
                termination_score = -1.0
        else:
            if hypothesis_score < 0.3:
                termination_score = 1.0
        clean_parse_score = 1 if clean_parse else -1.0
        total_score = hypothesis_score + termination_score + clean_parse_score
        return total_score

    def get_reasoning_reward(self, reasoning: str):
        reasoning_prompt_filled = self.judge_reasoning_prompt.replace(
            "[FUNCTION]",
            self.test_func_validated,
            "[FUNCTIONALITY]",
            self.description,
            "[HYPOTHESIS]",
            self.current_hypothesis,
            "[REASONING]",
            reasoning,
        )
        evaluation = self.judge_infer(reasoning_prompt_filled)
        # find rating: digit
        rating_match = re.search(r"rating:\s*(\d+)", evaluation.lower())
        reasoning_score = 0.0
        if rating_match:
            rating = int(rating_match.group(1))
            reasoning_score = rating / 9.0  # normalize to [0, 1]
        else:
            return 0.0
        return reasoning_score

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.runner is None:
            new_obs = {"role": "user", "content": "The test function code failed to execute, cannot run environment."}
            return BaseTextEnvStepOutput(
                observations=[new_obs],
                reward=0.0,
                done=True,
                metadata={"error": "Test function code failed to execute, cannot run environment."},
            )
        self.turns += 1
        if self.turn_kind == "reasoning":
            # TODO: Check first action
            if action is None:
                reward = 0.0
                pass
            else:
                clean_parse = False
                if action.lower().count("summary:") == 1:
                    decision, summary = (
                        action.lower().split("summary:")[0].strip(),
                        action.lower().split("summary:")[1].strip(),
                    )
                    hypothesis = summary
                    clean_parse = True
                else:
                    hypothesis = action.lower()
                    decision = "no"
                self.current_hypothesis = hypothesis
                reward = self.get_hypothesis_reward(decision, clean_parse)
            prompt = self.reasoning_prompt_filled.replace(
                "[PREV]", self.get_prev_results_str()
            ).replace("[HYPOTHESIS]", self.current_hypothesis)
            new_obs = {"role": "user", "content": prompt}
            self.turn_kind = "input"
            return BaseTextEnvStepOutput(
                observations=[new_obs],
                reward=reward,
                done=False,
                metadata={},
            )
        elif self.turn_kind == "input":
            self.previous_reasoning = action
            prompt = (
                self.input_prompt_filled.replace("[PREV]", self.get_prev_results_str())
                .replace("[HYPOTHESIS]", self.current_hypothesis)
                .replace(
                    "[REASONING]", action
                )  # TODO: check that this is reasoning output.
            )
            new_obs = {"role": "user", "content": prompt}
            reward = self.get_reasoning_reward(action)
            self.turn_kind = "reflection"
            return BaseTextEnvStepOutput(
                observations=[new_obs],
                reward=reward,
                done=False,
                metadata={},
            )
        elif self.turn_kind == "reflection":
            suggested_inputs = None
            options = response.strip().split("\n")
            for opt in options:
                if opt.count("Input:") == 1:
                    opt = opt.split("Input:")[1].strip()
                if opt.strip() != "":
                    suggested_inputs = opt
                    break
            if suggested_inputs is None:  # then empty string
                last_input_str = "You did not suggest any inputs. Do not do that again."
            # print(f"Suggested inputs: {suggested_inputs}")
            ret, err = self.runner.run_test_str(suggested_inputs)
            self.prev_results.append((suggested_inputs, ret, err))
            last_input_str = (
                "Input: " + suggested_inputs + f" => Output: {ret}, Error: {err}"
            )
            prompt = (
                self.reflection_prompt_filled.replace(
                    "[PREV]", self.get_prev_results_str()
                )
                .replace("[HYPOTHESIS]", self.current_hypothesis)
                .replace("[LAST_INPUTS]", last_input_str)
                .replace("[REASONING]", self.previous_reasoning)
            )
            new_obs = {"role": "user", "content": prompt}
            self.turn_kind = "reasoning"
            return BaseTextEnvStepOutput(
                observations=[new_obs],
                reward=0.1 if ret is not None else -0.1,
                done=False,
                metadata={},
            )




#################################### Code from the Examples Below: ################################################