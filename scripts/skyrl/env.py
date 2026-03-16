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
import numpy as np
from loguru import logger
import sys



# All rewards are generally in the range [-1, 1], but hypothesis scaling kinda changes that. 
PARSE_FAILURE_PENALTY = 0.5
MAX_LENGTH_PENALTY = 1.0 # Really go hardcore on Qwen3, it needs to shut up. 
HYPOTHESIS_SCALE = 2.0 # scale the hypothesis reward to ensure it is the dominant factor in the reward signal
VERBOSE = False


def neg(x):
    return -abs(x)

class RunTestFunc:
    """
    A class to run a test function defined in code. This differs from the one used in the main code, because of multiprocess spawn vs fork difference when using SkyRL.
    """

    def __init__(self, func_code: str, timeout=2.5):
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
        if success:
            pass
        else:
            raise RuntimeError("Failed to exec function code, cannot initialize RunTestFunc.") # This should never happen, the dataset should be pre-filtered to ensure exec safety and correctness, but we add this check just in case.


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

    def worker(self, func_code, args, queue):
        """Helper worker to run the function and put the result in a queue."""
        try:
            context = {"__builtins__": __builtins__}
            exec(func_code, context)
            func = context["test_func"]
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
            target=self.worker, args=(self.func_code, args, queue)
        )

        p.start()

        # Wait0
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
    base_url = "http://localhost:8000/v1/" # use "http://localhost:8000/v1" for vLLM servers, None for OpenAI API
    unsupervised = True


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
    - (arg0, arg1) [STOP] #(arg0, arg1) should be replaced with actual input values in your response and must be a valid python tuple. This is an example format for a two arg function. You should adjust the number of arguments as per the function definition.
    Now provide your suggested inputs below, do not say any reasoning or thinking. After saying the suggested input, say [STOP]
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
    Explanation (be extremely concise, no more than a sentence):
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
    Explanation: <your brief explanation here no more than a sentence>
    Rating: <a number from 0 to 9>
    [STOP]
    Now, provide your evaluation below. Be extremely concise in your explanation, no more than a sentence, and remember to say [STOP] at the end.
    Explanation: 
    """

    judge_code_generation_prompt = f"""
    You are given a function with the following header:
    [HEADER]
    The function accepts arguments and produces outputs, examples of which are as follows:
    [EXAMPLES]
    You are told the following is the functionality of the code:
    [FUNCTIONALITY]
    Given this, write a full python implementation of the function. Your implementation should be consistent with the provided examples and functionality description. Ensure your code is efficient and concise.
    Respond in the following format:
    Reasoning: <your concise reasoning here of how you will write the function, no more than a sentence>
    Code:
    ```python
    # your code implementation here
    ``` [STOP]
    Remember to follow the format strictly. Now provide your reasoning and code implementation below. Be concise in your reasoning, no more than a sentence, and ensure your code is correct and efficient.
    Reasoning: 
    """



    def length_penalty(self, response, threshold, penalty_rate, worst_threshold_multiplier=3):
        # penalize length over threshold at a rate of penalty_rate per token
        num_tokens = len(response.split())
        if num_tokens > threshold:
            penalty = penalty_rate * (num_tokens - threshold)
            worst_penalty = penalty_rate * (threshold * (worst_threshold_multiplier -1)) # worst case is when response is 3x the threshold, then we want to apply the max penalty.
            true_penalty = max(penalty / worst_penalty, MAX_LENGTH_PENALTY) # cap the penalty at MAX_LENGTH_PENALTY
            return neg(true_penalty)
        else:
            return 0.0
    
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
            self.judge_code_generation_prompt = self.judge_code_generation_prompt.replace(
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
            if LLMJudgeEnvConfig.base_url is not None:
                openai_api_key = "DUMMY"
                self.llm_judge_client = OpenAI(
                    api_key=openai_api_key,
                    base_url=LLMJudgeEnvConfig.base_url,
                )                
                self.model = "model"
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key is None:
                    raise ValueError("`OPENAI_API_KEY` must be set for Llm as a judge env")
                self.llm_judge_client = OpenAI(
                    api_key=openai_api_key
                )
                self.model = LLMJudgeEnvConfig.model
        except:
            self.runner = None


    def judge_infer(self, prompt, max_new_tokens=300):
        response = self.llm_judge_client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_new_tokens,
        )
        text = response.output_text
        if "[STOP]" in text:
            text = text.split("[STOP]")[0]
        if VERBOSE:
            logger.info(f"LLM Judge Prompt:\n{prompt}\nLLM Judge Response:\n{text}")
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
    
    def get_judge_prev_results_str(self):
        results_str = "```\n"
        for inp, out, err in self.prev_results:
            results_str += f">>> test_func{inp}\n"
            results_str += f"{out}"
        results_str += "\n```"
        return results_str
    
    def supervised_hypothesis_reward(self):
        hypothesis_prompt_filled = self.judge_hypothesis_prompt.replace(
            "[FUNCTION]",
            self.test_func_validated).replace(
            "[FUNCTIONALITY]",
            self.description).replace(
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
            return hypothesis_score
        else:
            return None
        
    def judge_code_generation(self):
        examples_str = self.get_judge_prev_results_str()
        code_generation_prompt_filled = self.judge_code_generation_prompt.replace(
            "[HEADER]", self.func_header).replace(
            "[EXAMPLES]", examples_str).replace(
            "[FUNCTIONALITY]", self.description)
        evaluation = self.judge_infer(code_generation_prompt_filled)
        if "```python" in evaluation:
            code = evaluation.split("```python")[1].split("```")[0].strip()
            return code
        else:
            return None

    def judge_code_evaluation(self, generated_code):
        try:
            test_runner = RunTestFunc(generated_code)
        except:
            return None
        exact_matches = []
        for inp, out, err in self.prev_results:
            gen_out, gen_err = test_runner.run_test_str(inp)
            if gen_err is not None and err is not None:
                exact_matches.append(1)
            elif gen_err is None and err is None:
                if gen_out == out:
                    exact_matches.append(1)
                else:
                    exact_matches.append(0)
            else:
                exact_matches.append(0)
        if len(exact_matches) == 0:
            return None
        code_score = sum(exact_matches) / len(exact_matches)
        return code_score
        
        
    def unsupervised_hypothesis_reward(self):
        generated_code = self.judge_code_generation()
        if generated_code is None:
            return None
        code_score = self.judge_code_evaluation(generated_code)
        return code_score

    def get_hypothesis_reward(self, done: bool):
        if LLMJudgeEnvConfig.unsupervised:
            hypothesis_score = self.unsupervised_hypothesis_reward()
        else:
            hypothesis_score = self.supervised_hypothesis_reward()
        if hypothesis_score is None:
            return 0.0
        termination_score = 0
        if done:
            if hypothesis_score > 0.7:
                termination_score = 1.0
            else:
                termination_score = -1.0
        else:
            if hypothesis_score < 0.3:
                termination_score = 1.0
        total_score = hypothesis_score + termination_score # this can technically exceed 1.0 but whatever bro.
        return total_score * HYPOTHESIS_SCALE

    def get_reasoning_reward(self, reasoning: str):
        reasoning_prompt_filled = self.judge_reasoning_prompt.replace(
            "[FUNCTION]",
            self.test_func_validated).replace(
            "[FUNCTIONALITY]",
            self.description).replace(
            "[HYPOTHESIS]",
            self.current_hypothesis).replace(
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
        if VERBOSE:
            logger.info(f"Step {self.turns}, Action: {action}, Turn Kind: {self.turn_kind}")
        if self.turns >= self.max_turns:
            new_obs = {"role": "user", "content": "Timeout"}
            return BaseTextEnvStepOutput(
                observations=[new_obs],
                reward=-1.0,
                done=True,
                metadata={"reason": "max_turns_reached"},
            )
        if self.turn_kind == "reasoning":
            clean_parse = False
            done = False
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
            if decision == "yes":
                done = True
            self.current_hypothesis = hypothesis
            if not clean_parse:
                reward = neg(PARSE_FAILURE_PENALTY)
            else:
                reward = 0.0 
            reward += self.get_hypothesis_reward(done) + self.length_penalty(action, threshold=100, penalty_rate=0.05)
            if VERBOSE:
                logger.info(f"Output: {action}\nHypothesis: {self.current_hypothesis}, Decision: {decision}, Reward: {reward}")
            prompt = self.reasoning_prompt_filled.replace(
                "[PREV]", self.get_prev_results_str()
            ).replace("[HYPOTHESIS]", self.current_hypothesis)
            new_obs = {"role": "user", "content": prompt}
            self.turn_kind = "input"
            return BaseTextEnvStepOutput(
                observations=[new_obs],
                reward=reward,
                done=done,
                metadata={},
            )
        elif self.turn_kind == "input":
            self.previous_reasoning = action
            prompt = (
                self.input_prompt_filled.replace("[PREV]", self.get_prev_results_str())
                .replace("[HYPOTHESIS]", self.current_hypothesis)
                .replace(
                    "[REASONING]", action
                )
            )
            new_obs = {"role": "user", "content": prompt}
            reward = self.get_reasoning_reward(action) + self.length_penalty(action, threshold=100, penalty_rate=0.05)
            self.turn_kind = "reflection"
            return BaseTextEnvStepOutput(
                observations=[new_obs],
                reward=reward,
                done=False,
                metadata={},
            )
        elif self.turn_kind == "reflection":
            suggested_inputs = None
            options = action.strip().split("\n")
            for opt in options:
                if opt.lower().count("input:") == 1:
                    opt = opt.split("Input:")[1].strip()
                    if opt.strip() != "":
                        suggested_inputs = opt
                        break
            if suggested_inputs is None:  # then empty string
                suggested_inputs = "INVALID INPUT"
                reward = neg(PARSE_FAILURE_PENALTY)
                ret, err = None, "Failed to parse input"
            else:
                reward = 0
                ret, err = self.runner.run_test_str(suggested_inputs)
                if err is None:
                    reward += 1/(self.max_turns) # Reward for successfully running the test function with the suggested input, encourages valid inputs.
                    # scale the reward to ensure it is never higher than the reward for getting a good hypothesis rating. 
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
                reward=reward + self.length_penalty(suggested_inputs, threshold=30, penalty_rate=0.5), # input should be super short.
                done=False,
                metadata={},
            )