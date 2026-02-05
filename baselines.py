from utils.lm_inference import HuggingFaceModel, OpenAIModel
import os
from utils import load_parameters, log_info, file_makedir
import click
from datasets import load_dataset
from data import RawLoaders
from eval import RunTestFunc
import pandas as pd
from tqdm import tqdm


def get_dataset(dataset_name, parameters=None):
    parameters = load_parameters()
    username = parameters["huggingface_repo_namespace"]
    dset = load_dataset(
        f"{username}/APIDiscoveryDataset", dataset_name, split="test"
    ).to_pandas()
    return dset


def get_save_paths(dataset_name, save_name):
    results_dir = f"results/{dataset_name}/"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(results_dir, f"{save_name}.jsonl"))
    return save_path


def zero_shot(
    func_code: str, examples: tuple, model, max_iterations=100, max_previous_results=10
):
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
        to_slice = (
            prev_results[-max_previous_results:]
            if max_previous_results is not None
            else prev_results
        )
        for inp, out, err in to_slice:
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


@click.command()
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def run_finetuned(dataset_name, model_name, save_name, override_gen):
    parameters = load_parameters()
    data_dir = parameters["data_dir"] + f"/final/{dataset_name}/"
    model_save_name = model_name.split("/")[-1].strip()
    save_name = model_save_name if save_name is None else save_name
    save_path = get_save_paths(dataset_name, save_name)
    file_makedir(save_path)
    if os.path.exists(save_path) and not override_gen:
        log_info(
            f"Output file {save_path} already exists, skipping generation. Run with override_gen=True to re-evaluate."
        )
    else:
        input_file = f"{data_dir}/test_filtered.csv"
        output_file = save_path
        df = RawLoaders.call_infer(
            run_name=save_name,
            dataset_name=dataset_name,
            split="test",
            input_file=input_file,
            output_file=output_file,
            output_column="predicted_description",
            max_new_tokens=300,
            model=model_name,
        )
        return


@click.command()
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def run_zeroshot(dataset_name, model_name, save_name, override_gen):
    model_save_name = model_name.split("/")[-1].strip()
    if save_name is None:
        save_name = model_save_name
    save_path = get_save_paths(dataset_name, save_name)
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
            dataset.iterrows(),
            total=len(dataset),
            desc=f"Evaluating {dataset_name}",
        ):
            test_func_str = row["test_func_validated"]
            true_description = row["description"]
            examples = row["train_inputs"]
            predicted_description, n_queries, concluded = zero_shot(
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
        log_info(f"Saved predictions to {save_path}")
    return save_path


@click.group()
def cli():
    pass


cli.add_command(run_finetuned, name="finetuned")
cli.add_command(run_zeroshot, name="zeroshot")

if __name__ == "__main__":
    cli()
