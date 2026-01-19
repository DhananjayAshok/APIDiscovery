from utils.lm_inference import HuggingFaceModel
from utils import log_warn, log_info
from data import RunTestFunc
from tqdm import tqdm
import pandas as pd
import os
from utils import load_parameters, log_info
import click
from datasets import load_dataset


def discover_function(func_code: str, examples: tuple, model, max_iterations=100):
    runner = RunTestFunc(func_code)
    header_start = func_code.index("def test_func(")
    header_end = func_code.index("\n", header_start)
    func_header = func_code[header_start:header_end]

    prev_results = []
    example_inputs = examples[0]
    example_outputs = examples[1]    
    for i, example_input in enumerate(example_inputs):
        input_str = example_input
        output, err = example_outputs[i], None
        prev_results.append((input_str, output, err))
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

Based on this, suggest an input to test the function with next.
The input should be valid Python tuples. 
Format Example: (arg0, arg1) [STOP]
Now provide your suggested inputs below and then say [STOP]
Suggested Input: 
"""
    reflection_prompt = f"""
You are given a Python function with the following header:
{func_header}
Your task is to try various inputs to discover what this function does.

So far, you have tried the following inputs: [PREV]
You then came up with the following running hypothesis: [HYPOTHESIS]
Finally, you just tried the following inputs: [LAST_INPUTS]

Based on this, can you conclude with very high confidence what the function does? If so, say YES and provide a concise description of its functionality.
Else, say NO and provide a revised hypothesis of what you think the function may do, and some guidance on how to test this further.
Format Example:
Hypothesis Conclusion: YES/NO |
Summary: <your summary or revised hypothesis here>
[STOP]

Now, provide your conclusion below, remember to say [STOP] after your summary.
Hypothesis C
"""
    for i in tqdm(range(max_iterations), desc="Function Discovery", leave=False):
        prev_results_str = get_prev_results_str()
        prompt = input_prompt.replace("[PREV]", prev_results_str).replace("[HYPOTHESIS]", hypothesis)
        response = model.generate(prompt, max_new_tokens=300, temperature=0.7)
        #print(response)
        #print("-----")
        suggested_inputs = [response.strip()]
        last_inputs = []
        for inp_str in suggested_inputs:
            ret, err = runner.run_test_str(inp_str)
            prev_results.append((inp_str, ret, err))
            last_inputs.append(inp_str)
        last_input_str = "\n".join(suggested_inputs)
        reflection = reflection_prompt.replace("[PREV]", prev_results_str).replace("[HYPOTHESIS]", hypothesis).replace("[LAST_INPUTS]", last_input_str)
        reflection_response = model.generate(reflection, max_new_tokens=300, temperature=0.7)
        #print(reflection_response)
        #print("=====")
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

class Evaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = HuggingFaceModel(model_name=model_name)

    def evaluate(self, hypothesis: str, true_description: str) -> float:
        prompt = f"""
You are given a function description and a hypothesized description of what the function does.
Your task is to rate how accurate the hypothesized description is compared to the true description on a scale from 1 to 5, where 1 means "completely inaccurate" and 5 means "completely accurate".
Example:
True Function Description: This function takes a list of integers and returns True if there are any two integers in the list that sum to zero, otherwise it returns False.
Hypothesized Description: This function checks if there are two numbers in the list that add up to zero.
Explanation: The hypothesized description accurately captures the functionality of the true description.
Rating: 5 [STOP]

True Function Description: calulates the nth fibonacci number
Hypothesized Description: This function computes the factorial of a number.
Explanation: The hypothesized description is incorrect as the fibonacci sequence and factorial are different mathematical concepts.
Rating: 1 [STOP]

Now, provide your rating for the following description only. You absolutely must follow the format shown in the examples above and no matter what, you must provide a rating between 1 and 5.
True Function Description: {true_description}
Hypothesized Description: {hypothesis}
Explanation (very short): The hypothesized description is
"""
        response = self.model.generate(prompt.strip(), max_new_tokens=150)
        response = response.strip().lower()
        if response.count("rating:") == 1:
            response = response.split("rating:")[1].strip()
        else:
            log_warn("Could not find 'Rating:' (or found multiple) in model response: " + response)
            return None
        if response.isdigit():
            rating = int(response)
            if 1 <= rating <= 5:
                return rating
        else:
            log_warn("Could not parse rating from model response: " + response)
        return None

def get_dataset(dataset_name, parameters=None):
    parameters = load_parameters()
    username = parameters["huggingface_repo_namespace"]
    dset = load_dataset(f"{username}/APIDiscoveryDataset", dataset_name, split="test").to_pandas()
    return dset

def score_dataset(dataset_name, model_name):
    dataset = get_dataset(dataset_name)
    evaluator = Evaluator(model_name="Qwen/Qwen3-32B")
    model = HuggingFaceModel(model_name=model_name)    
    columns = ["test_func_validated", "true_description", "n_queries", "concluded", "predicted_description", "score"]
    data = []
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {dataset_name}"):
        test_func_str = row["test_func_validated"]
        true_description = row["description"]
        examples = (row["train_inputs"], row["train_outputs"])
        predicted_description, n_queries, concluded = discover_function(test_func_str, examples, model)
        score = evaluator.evaluate(predicted_description, true_description)
        data.append([test_func_str, true_description, n_queries, concluded, predicted_description, score])
    df = pd.DataFrame(data=data, columns=columns)
    parameters = load_parameters()
    save_path = "results/"
    os.makedirs(save_path, exist_ok=True)
    model_save_name = model_name.split("/")[-1].strip()
    df.to_csv(f"results/{dataset_name}_{model_save_name}.csv", index=False)
    avg_n_queries = df["n_queries"].mean()
    avg_score = df["score"].mean()
    perc_concluded = df["concluded"].mean()
    log_info(f"n_queries: {avg_n_queries}, concluded: {round(perc_concluded* 100, 2)}, score: {avg_score}")
    log_info(df.groupby("concluded")["score"].mean())
    log_info(df[["n_queries", "score"]].mean())    

    
@click.command()
@click.option("--dataset_name", type=str, required=True)
@click.option("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
def do(dataset_name, model_name):
    score_dataset(dataset_name, model_name)
    
if __name__ == "__main__":
    do()
