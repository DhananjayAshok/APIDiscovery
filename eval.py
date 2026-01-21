from utils.lm_inference import HuggingFaceModel
from utils import log_warn, log_info
from data import RunTestFunc
from tqdm import tqdm
import pandas as pd
import os
from utils import load_parameters, log_info, file_makedir
import click
from datasets import load_dataset
import subprocess


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
Format Example for a two arg function: 
Reasoning: Loosely describe your intended input, and the properties it satisfies. How does this input help test the hypothesis? What is the expected output?
Suggested Input:
(arg0, arg1) [STOP] #(arg0, arg1) should be replaced with actual input values in your response.
Now provide your reasoning and suggested inputs below and then say [STOP]
Reasoning:"""

    reflection_prompt = f"""
You are given a Python function with the following header:
{func_header}
Your task is to try various inputs to discover what this function does.

So far, you have tried the following inputs: [PREV]
You then came up with the following running hypothesis: [HYPOTHESIS]
Finally, you just tried the following inputs: [LAST_INPUTS]

Based on this, can you conclude with very high confidence what the function does? If the function did not perform as you expected, the answer is likely no. If you think it is yes, then say YES and provide a concise description of its functionality.
Else, say NO and provide a revised hypothesis of what you think the function may do, and some guidance on how to test this further.
Format Example:
Hypothesis Conclusion: YES/NO |
Summary: <your summary or revised hypothesis here>
[STOP]

Now, provide your conclusion below, remember to say [STOP] after your summary.
Hypothesis C"""
    for i in tqdm(range(max_iterations), desc="Function Discovery", leave=False):
        prev_results_str = get_prev_results_str()
        prompt = input_prompt.replace("[PREV]", prev_results_str).replace("[HYPOTHESIS]", hypothesis)
        response = model.generate(prompt, max_new_tokens=300)
        #print(response)
        #print("-----")
        reasoning_part = None
        suggest_input_part = None
        if response.count("Suggested Input:") == 1:
            reasoning_part, suggest_input_part = response.split("Suggested Input:")
        else:
            log_warn("Could not find 'Suggested Input:' (or found multiple) in model response: " + response)
        if suggest_input_part is None:
            last_input_str = f"ERROR, YOU RETURNED THE TEXT: '{response.strip()}'. This does not follow the required format. You must provide a 'Suggested Input:' section."
        else:
            suggested_inputs = suggest_input_part.strip().split("[STOP]")[0].strip()
            ret, err = runner.run_test_str(suggested_inputs)
            prev_results.append((suggested_inputs, ret, err))
            last_input_str = "Input: " + suggested_inputs + f". Expectation: {reasoning_part} => Output: {ret}, Error: {err}"
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

eval_prompt = f"""
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
True Function Description: [TRUE]
Hypothesized Description: [HYPOTHESIS]
Explanation (very short): The hypothesized description is
"""    
def score_predictions(dataset_name, save_name):
    parameters = load_parameters()
    model = parameters["evaluation_model_name"]
    input_file = get_save_path(dataset_name, save_name)
    output_file = input_file.replace(".jsonl", "_scored.jsonl")
    df = pd.read_json(input_file, lines=True)
    def get_score_prompt(row):
        prompt_filled = eval_prompt.replace("[TRUE]", row["true_description"]).replace("[HYPOTHESIS]", row["predicted_description"])
        return prompt_filled
    def parse_score(output):
        response = output.strip().lower()
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
    df["score_prompt"] = df.apply(get_score_prompt, axis=1)      
    df.to_json(input_file, orient="records", lines=True)  
    open_ai_batch_name = ""
    if "gpt" in model:
        open_ai_batch_name = f"judge-{dataset_name}-{save_name}-{model}"
    openaibatch_str = "-n " + open_ai_batch_name if open_ai_batch_name != "" else ""
    command_string = f"bash scripts/infer.sh -i {input_file} -o {output_file} -m {model} -c score_prompt -d score_output -t 300 -g judge {openaibatch_str}"
    log_info(f"Generating scores with command: {command_string}")
    subprocess.run(command_string, shell=True, check=True)
    try:
        df = pd.read_json(output_file, lines=True)
        if "output_logits" in df.columns:
            df.drop("output_logits", axis=1, inplace=True)
        df["score"] = df["score_output"].apply(parse_score)            
        df.to_json(output_file, orient="records", lines=True)
        return df
    except:
        log_warn(f"Output file {output_file} not found after inference command.")
    return None

def get_dataset(dataset_name, parameters=None):
    parameters = load_parameters()
    username = parameters["huggingface_repo_namespace"]
    dset = load_dataset(f"{username}/APIDiscoveryDataset", dataset_name, split="test").to_pandas()
    return dset

def get_save_path(dataset_name, save_name):
    results_dir = f"results/{dataset_name}/"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{save_name}.jsonl")
    return save_path

def run_eval_on_dataset(dataset_name, model_name, save_name=None, override=False):
    model_save_name = model_name.split("/")[-1].strip()
    if save_name is None:
        save_name = model_save_name    
    save_path = get_save_path(dataset_name, save_name)
    file_makedir(save_path)
    if os.path.exists(save_path) and not override:
        log_info(f"Evaluation file {save_path} already exists, skipping evaluation. Run with override=True to re-evaluate.")
    else:
        dataset = get_dataset(dataset_name)
        model = HuggingFaceModel(model_name=model_name)    
        columns = ["test_func_validated", "true_description", "n_queries", "concluded", "predicted_description"]
        data = []
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {dataset_name}"):
            test_func_str = row["test_func_validated"]
            true_description = row["description"]
            examples = (row["train_inputs"], row["train_outputs"])
            predicted_description, n_queries, concluded = discover_function(test_func_str, examples, model)
            data.append([test_func_str, true_description, n_queries, concluded, predicted_description])
        df = pd.DataFrame(data=data, columns=columns)
        df.to_json(save_path, orient="records", lines=True)
    scored_df = score_predictions(dataset_name, save_name)
    if scored_df is not None:
        avg_n_queries = scored_df["n_queries"].mean()
        avg_score = scored_df["score"].mean()
        perc_concluded = scored_df["concluded"].mean()
        log_info(f"n_queries: {avg_n_queries}, concluded: {round(perc_concluded* 100, 2)}, score: {avg_score}")
        log_info(df.groupby("concluded")["score"].mean())
        log_info(df[["n_queries", "score"]].mean())

    
@click.command()
@click.option("--dataset_name", type=str, required=True)
@click.option("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
@click.option("--save_name", type=str, default=None)
@click.option("--override", is_flag=True, default=False)
def do(dataset_name, model_name, save_name, override):
    run_eval_on_dataset(dataset_name, model_name, save_name, override)
    
if __name__ == "__main__":
    do()
