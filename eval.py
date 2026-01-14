from utils.lm_inference import HuggingFaceModel
from utils import log_warn, log_info
from data import RunTestFunc
from tqdm import tqdm


def discover_function(func_code: str, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_iterations=100):
    model = HuggingFaceModel(model_name=model_name)
    runner = RunTestFunc(func_code)
    header_start = func_code.index("def test_func(")
    header_end = func_code.index("\n", header_start)
    func_header = func_code[header_start:header_end]
    prev_results = []
    concluded = False
    def get_prev_results_str():
        if not prev_results:
            return "[]"
        results_str = "[\n"
        for inp, out, err in prev_results:
            results_str += f"  Input: {inp} => Output: {out}, Error: {err}\n"
        results_str += "]"
        return results_str
    hypothesis = "Not yet formed"
    input_prompt = f"""
You are given a Python function with the following header:
{func_header}
Your task is to try various inputs to discover what this function does.

So far, you have tried the following inputs: [PREV]
You then came up with the following running hypothesis: [HYPOTHESIS]

Based on this, suggest a short list of inputs to test the function with next.
Provide each function calls arguments on a separate line. Each line should be valid Python tuples. 
Format Example:
- (arg0, arg1) # if the function takes exactly two arguments
Now provide your suggested inputs below and when you are done with the list, say [STOP]
Suggested Inputs:

"""
    reflection_prompt = f"""
You are given a Python function with the following header:
{func_header}
Your task is to try various inputs to discover what this function does.

So far, you have tried the following inputs: [PREV]
You then came up with the following running hypothesis: [HYPOTHESIS]
Finally, you just tried the following inputs: [LAST_INPUTS]

Based on this, can you conclude what the function does? If so, say YES and provide a concise description of its functionality.
Else, say NO and provide a revised hypothesis of what you think the function may do, and some guidance on how to test this further.
Format Example:
Hypothesis Conclusion: YES/NO |
Summary: <your summary or revised hypothesis here>
[STOP]

Now, provide your conclusion below:
Hypothesis Conclusion: 
"""
    for i in tqdm(range(max_iterations), desc="Function Discovery"):
        prev_results_str = get_prev_results_str()
        prompt = input_prompt.replace("[PREV]", prev_results_str).replace("[HYPOTHESIS]", hypothesis)
        response = model.generate(prompt, max_new_tokens=300, temperature=0.7)
        print(response)
        print("-----")
        suggested_inputs = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
                suggested_inputs.append(line)
        if not suggested_inputs:
            log_warn("No suggested inputs found, stopping discovery. Output text was: \n" + response)
            break
        last_inputs = []
        for inp_str in suggested_inputs:
            ret, err = runner.run_test_str(inp_str)
            prev_results.append((inp_str, ret, err))
            last_inputs.append(inp_str)
        last_input_str = "\n".join(suggested_inputs)
        reflection = reflection_prompt.replace("[PREV]", prev_results_str).replace("[HYPOTHESIS]", hypothesis).replace("[LAST_INPUTS]", last_input_str)
        reflection_response = model.generate(reflection, max_new_tokens=300, temperature=0.7)
        print(reflection_response)
        print("=====")
        if "summary:" in reflection_response.lower():
            hypothesis = reflection_response.lower().split("summary:",1)[1].strip()
        else:
            hypothesis = reflection_response.lower()
        if "yes" in reflection_response.lower().split("|")[0]:
            concluded = True
            break
        else:
            pass
    return hypothesis, runner.access_counter, concluded
    
    
if __name__ == "__main__":
    # Example usage
    func_code = """
from typing import List    
def validate_input_args(arg0: List[float], arg1: float) -> None:
    if not isinstance(arg0, list):
        raise TypeError("arg0 must be a list")
    for item in arg0:
        if not isinstance(item, float):
            raise TypeError("All elements in arg0 must be floats")
    if not isinstance(arg1, float):
        raise TypeError("arg1 must be a float")
    return


def test_func(arg0: List[float], arg1: float) -> bool:
    validate_input_args(arg0, arg1)
    threshold = arg1
    numbers = arg0


    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
"""
    hypothesis, n_queries, concluded = discover_function(func_code, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_iterations=50)
    log_info(f"Final Hypothesis: {hypothesis}")
    log_info(f"Total Queries Used: {n_queries}")
    log_info(f"Concluded: {concluded}")