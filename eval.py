from utils.lm_inference import HuggingFaceModel, OpenAIModel
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
    example_outputs = []
    for example in examples:
        example_outputs.append(runner.run_test_str(example))

    for i, example_input in enumerate(examples):
        input_str = example_input
        output, err = example_outputs[i]
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
    reasoning_prompt = f"""
You are given a Python function with the following header:
{func_header}
Your task is to try various inputs to discover what this function does.

So far, you have tried the following inputs: [PREV]
You then came up with the following running hypothesis: [HYPOTHESIS]

Based on this, what kind of input will you use to test the function with next? Very briefly describe your next intended input only, and the properties it satisfies. How does this input help test the hypothesis? What is the expected output? Be extremely concise and short. 
Your response should be extremely short and concise, just a few sentences. After the response, say [STOP]
Now provide your reasoning below and then say [STOP]
Reasoning:"""

    input_prompt = f"""
You are given a Python function with the following header:
{func_header}
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
{func_header}
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
Hypothesis Conclusion: """
    for i in tqdm(range(max_iterations), desc="Function Discovery", leave=False):
        prev_results_str = get_prev_results_str()
        prompt = reasoning_prompt.replace("[PREV]", prev_results_str).replace(
            "[HYPOTHESIS]", hypothesis
        )
        response = model.generate(prompt, max_new_tokens=300)
        reasoning = response.split("[STOP]")[0].strip()
        prompt = (
            input_prompt.replace("[PREV]", prev_results_str)
            .replace("[HYPOTHESIS]", hypothesis)
            .replace("[REASONING]", reasoning)
        )
        response = model.generate(prompt, max_new_tokens=300)
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
        ret, err = runner.run_test_str(suggested_inputs)
        prev_results.append((suggested_inputs, ret, err))
        last_input_str = (
            "Input: " + suggested_inputs + f" => Output: {ret}, Error: {err}"
        )
        reflection = (
            reflection_prompt.replace("[PREV]", prev_results_str)
            .replace("[HYPOTHESIS]", hypothesis)
            .replace("[LAST_INPUTS]", last_input_str)
            .replace("[REASONING]", reasoning)
        )
        reflection_response = model.generate(reflection, max_new_tokens=300)
        if reflection_response.lower().count("summary:") == 1:
            decision, summary = (
                reflection_response.lower().split("summary:")[0].strip(),
                reflection_response.lower().split("summary:")[1].strip(),
            )
            hypothesis = summary
        else:
            hypothesis = reflection_response.lower()
            decision = "no"
        if False:
            print(f"Iteration {i+1}:")
            print(f"Reasoning: {reasoning}")
            print(f"Suggested inputs: {suggested_inputs}")
            print(f"Function output: {ret}, Error: {err}")
            print(f"Reflection response: {reflection_response}")
        if "yes" in decision:
            concluded = True
            break
        else:
            pass
    return hypothesis, runner.access_counter, concluded


eval_prompt = f"""
You are given a function description and a hypothesized description of what the function does.
Your task is to rate how accurate the hypothesized description is compared to the true description on a scale from 1 to 5, where 1 means "completely inaccurate" and 5 means "completely accurate".
First, provide an extremely brief explanation (1 sentence) of why you gave that rating. Then, provide your rating in the format "Rating: X" where X is an integer between 1 and 5.
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
Explanation (very short):"""


def score_predictions(dataset_name, save_name, save_path, evaluation_path):
    parameters = load_parameters()
    model = parameters["evaluation_model_name"]
    df = pd.read_json(save_path, lines=True)

    def get_score_prompt(row):
        prompt_filled = eval_prompt.replace("[TRUE]", row["true_description"]).replace(
            "[HYPOTHESIS]", row["predicted_description"]
        )
        return prompt_filled

    def parse_score(output):
        response = output.strip().lower()
        if response.count("rating:") == 1:
            response = response.split("rating:")[1].strip()
        else:
            log_warn(
                "Could not find 'Rating:' (or found multiple) in model response: "
                + response
            )
            return None
        if response.isdigit():
            rating = int(response)
            if 1 <= rating <= 5:
                return rating
        else:
            log_warn("Could not parse rating from model response: " + response)
        return None

    df["score_prompt"] = df.apply(get_score_prompt, axis=1)
    df.to_json(save_path, orient="records", lines=True)
    open_ai_batch_name = ""
    if "gpt" in model:
        open_ai_batch_name = f"judge-{dataset_name}-{save_name}-{model}"
    openaibatch_str = "-n " + open_ai_batch_name if open_ai_batch_name != "" else ""
    command_string = f"bash scripts/infer.sh -i {save_path} -o {evaluation_path} -m {model} -c score_prompt -d score_output -t 300 -g judge {openaibatch_str}"
    log_info(f"Generating scores with command: {command_string}")
    subprocess.run(command_string, shell=True, check=True)
    try:
        df = pd.read_json(evaluation_path, lines=True)
        if isinstance(df["score_output"][0], list):
            df["score_output"] = df["score_output"].apply(
                lambda x: x[0] if len(x) > 0 else ""
            )
        if "output_logits" in df.columns:
            df.drop("output_logits", axis=1, inplace=True)
        df["score"] = df["score_output"].apply(parse_score)
        df.to_json(evaluation_path, orient="records", lines=True)
        return df
    except:
        log_warn(
            f"Output file {evaluation_path} not found after inference command. This can happen for openai inference. Run the script again when the batch is done. "
        )
    return None


def get_dataset(dataset_name, parameters=None):
    parameters = load_parameters()
    username = parameters["huggingface_repo_namespace"]
    dset = load_dataset(
        f"{username}/APIDiscoveryDataset", dataset_name, split="test_"
    ).to_pandas()
    return dset


def get_save_paths(dataset_name, save_name):
    results_dir = f"results/{dataset_name}/"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(results_dir, f"{save_name}.jsonl"))
    evaluation_path = os.path.abspath(
        os.path.join(results_dir, f"{save_name}_scored.jsonl")
    )
    return save_path, evaluation_path


def run_eval_on_dataset(
    dataset_name, model_name, save_name=None, override_gen=False, override_eval=False
):
    model_save_name = model_name.split("/")[-1].strip()
    if save_name is None:
        save_name = model_save_name
    save_path, evaluation_path = get_save_paths(dataset_name, save_name)
    file_makedir(save_path)
    if os.path.exists(save_path) and not override_gen:
        log_info(
            f"Output file {save_path} already exists, skipping generation. Run with override_gen=True to re-evaluate."
        )
    else:
        dataset = get_dataset(dataset_name)
        if "gpt" in model_name:
            model = OpenAIModel(model_name=model_name)
        else:
            model = HuggingFaceModel(model_name=model_name)
        columns = [
            "test_func_validated",
            "true_description",
            "n_queries",
            "concluded",
            "predicted_description",
        ]
        data = []
        for i, row in tqdm(
            dataset[:5].iterrows(),
            total=len(dataset),
            desc=f"Evaluating {dataset_name}",
        ):
            test_func_str = row["test_func_validated"]
            true_description = row["description"]
            examples = row["train_inputs"]
            predicted_description, n_queries, concluded = discover_function(
                test_func_str, examples, model
            )
            data.append(
                [
                    test_func_str,
                    true_description,
                    n_queries,
                    concluded,
                    predicted_description,
                ]
            )
        df = pd.DataFrame(data=data, columns=columns)
        df.to_json(save_path, orient="records", lines=True)
    if os.path.exists(evaluation_path) and not override_eval:
        log_info(
            f"Evaluation file {evaluation_path} already exists, skipping evaluation. Run with override_eval=True to re-evaluate."
        )
        scored_df = pd.read_json(evaluation_path, lines=True)
    else:
        if os.path.exists(evaluation_path):
            os.remove(evaluation_path)
        scored_df = score_predictions(
            dataset_name,
            save_name,
            save_path=save_path,
            evaluation_path=evaluation_path,
        )
    if scored_df is not None:
        avg_n_queries = scored_df["n_queries"].mean()
        avg_score = scored_df["score"].mean()
        perc_concluded = scored_df["concluded"].mean()
        log_info(
            f"n_queries: {avg_n_queries}, concluded: {round(perc_concluded* 100, 2)}, score: {avg_score}"
        )
        log_info(scored_df.groupby("concluded")["score"].mean())
        log_info(scored_df[["n_queries", "score"]].mean())


@click.command()
@click.option("--dataset_name", type=str, required=True)
@click.option("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
@click.option("--save_name", type=str, default=None)
@click.option("--override_gen", is_flag=True, default=False)
@click.option("--override_eval", is_flag=True, default=False)
def do(dataset_name, model_name, save_name, override_gen, override_eval):
    run_eval_on_dataset(
        dataset_name, model_name, save_name, override_gen, override_eval
    )


if __name__ == "__main__":
    do()
