from utils.lm_inference import HuggingFaceModel, OpenAIModel
import os
from utils import (
    load_parameters,
    log_info,
    file_makedir,
    log_warn,
    log_error,
    get_test_func_header,
    call_infer,
    get_zeroshot_starting_details,
    get_prev_results_str,
    RunTestFunc,
)
from eval import rectify_description, get_output_save_name, get_code_save_name, get_input_save_name
import click
from load_data import get_dataset
import pandas as pd
from tqdm import tqdm


def get_save_paths(dataset_name, save_name):
    results_dir = f"results/{dataset_name}/"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(results_dir, f"{save_name}.jsonl"))
    return save_path


def zero_shot(
    func_code: str, examples, model, max_iterations=100, max_previous_results=10
):
    reasoning_prompt, runner, prev_results, func_header = get_zeroshot_starting_details(
        func_code, examples
    )
    concluded = False
    hypothesis = "Not yet formed"
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
    steps = []
    for i in tqdm(range(max_iterations), desc="Function Discovery", leave=False):
        prev_results_str = get_prev_results_str(prev_results, max_previous_results)
        prompt = reasoning_prompt.replace("[PREV]", prev_results_str).replace(
            "[HYPOTHESIS]", hypothesis
        )
        response = model.generate(prompt, max_new_tokens=300)
        reasoning = response.split("[STOP]")[0].strip()
        steps.append({"input": prompt, "output": reasoning})
        prompt = (
            input_prompt.replace("[PREV]", prev_results_str)
            .replace("[HYPOTHESIS]", hypothesis)
            .replace("[REASONING]", reasoning)
        )
        response = model.generate(prompt, max_new_tokens=300)
        steps.append({"input": prompt, "output": response})
        suggested_inputs = None
        options = response.strip().split("\n")
        for opt in options:
            if opt.count("input:") == 1 or opt.count("Input:") == 1:
                if opt.count("input:") == 1:
                    opt = opt.split("input:")[1].strip()
                else:
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
            "Input: " + suggested_inputs
            if suggested_inputs is not None
            else "None" + f" => Output: {ret}, Error: {err}"
        )
        reflection = (
            reflection_prompt.replace("[PREV]", prev_results_str)
            .replace("[HYPOTHESIS]", hypothesis)
            .replace("[LAST_INPUTS]", last_input_str)
            .replace("[REASONING]", reasoning)
        )
        reflection_response = model.generate(reflection, max_new_tokens=300)
        steps.append({"input": reflection_prompt, "output": reflection_response})
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
            print(f"Iteration {i + 1}:")
            print(f"Reasoning: {reasoning}")
            print(f"Suggested inputs: {suggested_inputs}")
            print(f"Function output: {ret}, Error: {err}")
            print(f"Reflection response: {reflection_response}")
        if "yes" in decision:
            concluded = True
            break
        else:
            pass
    return hypothesis, runner.access_counter, concluded, steps, prev_results


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
        input_file = f"{data_dir}/test_filtered.jsonl"
        output_file = save_path
        call_infer(
            run_name=save_name,
            dataset_name=dataset_name,
            split="test",
            input_file=input_file,
            output_file=output_file,
            input_column="direct_prompt",
            output_column="predicted_description",
            max_new_tokens=300,
            model=model_name,
            parameters=parameters,
            ignore_checkpoint=override_gen,
        )
        if os.path.exists(output_file):
            df = pd.read_json(output_file, lines=True)
            df["predicted_description_clean"] = df["predicted_description"].apply(rectify_description)
            df.to_json(output_file, orient="records", lines=True)
            log_info(f"Saved predictions to {output_file}")
        else:
            log_warn(
                f"Output file {output_file} was not created. This can happen with OpenAI inference. Run again when batch is done."
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
            "steps",
            "all_examples",
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
            predicted_description, n_queries, concluded, steps, all_examples = (
                zero_shot(test_func_str, examples, model)
            )
            repr_examples = repr(all_examples)
            data.append(
                [
                    test_func_str,
                    true_description,
                    n_queries,
                    concluded,
                    predicted_description,
                    steps,
                    repr_examples,
                ]
            )
        df = pd.DataFrame(data=data, columns=columns)
        df["predicted_description_clean"] = df["predicted_description"].apply(rectify_description)
        for column in df:
            dataset[column] = df[column]
        dataset.to_json(save_path, orient="records", lines=True)
        log_info(f"Saved predictions to {save_path}")
    return save_path


code_prediction_prompt = """
You are an expert programmer. Your goal is to create a Python function called `test_func` that matches the following description:
[DESCRIPTION]

The header of the function must be:
[HEADER]

The function must satisfy the following input-output examples:
[EXAMPLES]

Now, write the complete code for the function `test_func` that meets the above requirements. You should structure your response as follows:
Reasoning: <any brief thinking or reasoning you want to do before writing the code. This must be extremely brief and concise, just a sentence or two at most>
Code:
```python
<your code here>
```
[STOP] # make sure to include the [STOP] token at the end of your response to indicate that you have finished writing the code.
Now, provide your reasoning and code below, and remember to end with [STOP].
Reasoning: 
"""


def run_eval_code(
    model_name: str,
    dataset_name: str,
    save_name: str,
    override_gen: bool,
    input_file: str,
    output_file: str,
    intermediate_file: str,
    df: pd.DataFrame,
    prompt_column: str,
    output_column: str = "predicted_code_output",
    max_new_tokens: int = 600,
):
    """
    Common function for running code prediction evaluation.
    Takes a DataFrame with prompts, runs inference, extracts code, and saves results.
    """
    df.to_json(input_file, orient="records", lines=True)
    df = call_infer(
        run_name=save_name,
        dataset_name=dataset_name,
        split="test",
        input_file=input_file,
        output_file=intermediate_file,
        input_column=prompt_column,
        output_column=output_column,
        max_new_tokens=max_new_tokens,
        model=model_name,
        parameters=load_parameters(),
        ignore_checkpoint=override_gen,
    )
    if not os.path.exists(intermediate_file):
        log_warn(
            f"Intermediate file {intermediate_file} was not created. This can happen with OpenAI inference. Run again when batch is done."
        )
        return df
    df = pd.read_json(intermediate_file, orient="records", lines=True)
    parse_errors = 0

    def extract_code(row):
        response = row[output_column]
        if isinstance(response, list):
            response = response[0]
        if "description" in row:
            true_description = row["description"]
        else:
            true_description = row["true_description"]
        if "[STOP]" in response:
            code_part = response.split("[STOP]")[0]
        else:
            code_part = response
        if "```python" in code_part and "```" in code_part:
            code = code_part.split("```python")[1].split("```")[0].strip()
            return code
        else:
            return None

    df["predicted_code"] = None
    for i, row in df.iterrows():
        code = extract_code(row)
        if code is not None:
            df.at[i, "predicted_code"] = code
        else:
            parse_errors += 1
    df.to_json(output_file, orient="records", lines=True)
    log_info(
        f"Saved predicted code to {output_file} | Parse errors: {parse_errors}/{len(df)}"
    )
    return df


def get_all_examples_str(row):
    if "all_examples" in row:
        examples = eval(row["all_examples"])
        examples = [
            f"Input: {examples[i][0]} => Output: {examples[i][1]}, Error: {examples[i][2]}"
            for i in range(len(examples))
        ]
    else:
        examples = row["train_inputs"]
        if isinstance(examples, str):
            examples = eval(examples)
        test_func_code = row["test_func_validated"]
        example_outputs = []
        try:
            runner = RunTestFunc(test_func_code)
            for example in examples:
                example_outputs.append(runner.run_test_str(example))
        except:
            log_warn(
                f"Could not run test function for row {row}. This should never happen."
            )
            return None
        examples = [
            f"Input: {examples[i]} => Output: {example_outputs[i][0]}, Error: {example_outputs[i][1]}"
            for i in range(len(examples))
        ]
    examples_str = "\n".join(examples)
    return examples_str


def get_code_model(model_name):
    parameters = load_parameters()
    code_generation_model = parameters["code_generation_model_name"]
    if code_generation_model == "self":
        code_generation_model = model_name
    return code_generation_model


def do_predict_code(
    model_name, dataset_name, save_name, override_gen, prediction_column, load_name=None
):
    if load_name is None:
        load_name = save_name
    prediction_file = get_save_paths(dataset_name, load_name)
    save_name = get_code_save_name(save_name)
    code_generation_model = get_code_model(model_name)
    output_file = get_save_paths(dataset_name, save_name)
    if os.path.exists(output_file) and not override_gen:
        log_info(
            f"Output file {output_file} already exists, skipping generation. Run with override_gen=True to re-evaluate."
        )
        return
    input_file = output_file.replace(".jsonl", "_input.jsonl")
    intermediate_file = output_file.replace(".jsonl", "_intermediate.jsonl")
    df = pd.read_json(prediction_file, orient="records", lines=True)

    def make_code_prompt(row):
        true_description = None
        if "true_description" in row:
            true_description = row["true_description"]
        else:
            true_description = row["description"]
        predicted_description = row["predicted_description_clean"]
        test_func_str = row["test_func_validated"]
        func_header = get_test_func_header(test_func_str)
        examples_str = get_all_examples_str(row)
        if prediction_column == "prediction":
            use_description = predicted_description
        elif prediction_column == "true":
            use_description = true_description
        else:
            log_error(
                f"Invalid prediction column: {prediction_column}. Must be either 'prediction' or 'true'."
            )

        prompt = (
            code_prediction_prompt.replace("[DESCRIPTION]", use_description)
            .replace("[HEADER]", func_header)
            .replace("[EXAMPLES]", examples_str)
        )
        return prompt

    df["code_prediction_prompt"] = df.apply(make_code_prompt, axis=1)
    run_eval_code(
        model_name=code_generation_model,
        dataset_name=dataset_name,
        save_name=save_name,
        override_gen=override_gen,
        input_file=input_file,
        output_file=output_file,
        intermediate_file=intermediate_file,
        df=df,
        prompt_column="code_prediction_prompt",
        output_column="predicted_code_output",
        max_new_tokens=600,
    )


@click.command()
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def predict_code(model_name, dataset_name, save_name, override_gen):
    do_predict_code(
        model_name,
        dataset_name,
        save_name,
        override_gen,
        prediction_column="prediction",
    )


output_prediction_prompt = """
You are an expert programmer. Your goal is to reason about a function that matches the following description:
[DESCRIPTION]

The function satisfies the following input-output examples:
[EXAMPLES]

Now, here is a new input to the function: 
[INPUT]
What is the expected output of the function on this input?
Your response should follow the format below:
Reasoning: <any brief thinking or reasoning you want to do before giving the output. This must be extremely brief and concise, just a sentence or two at most>
Expected Output: <the expected output here. This must be a valid python expression that will match the output of the function when evaluated.>
[STOP] # make sure to include the [STOP] token at the end of your response to indicate that you have finished giving the expected output.
Now, provide your reasoning and expected output below, and remember to end with [STOP].
Reasoning: 
"""

input_prediction_prompt = """
You are an expert programmer. Your goal is to reason about a function that matches the following description:
[DESCRIPTION]

The header of the function is:
[HEADER]

The function satisfies the following input-output examples:
[EXAMPLES]

Now, here is a new output from the function:
[OUTPUT]
What is an input that would produce this output when passed through the function? Your response should follow the format below:
Reasoning: <any brief thinking or reasoning you want to do before giving the input. This must be extremely brief and concise, just a sentence or two at most>
Suggested Input: <the suggested input here. This must be a valid python tuple that can be evaluated and inputed into the function with the correct number of arguments. For example, if the function takes two arguments, your suggested input should be a tuple like (arg0, arg1) with appropriate values for arg0 and arg1.>
[STOP] # make sure to include the [STOP] token at the end of your response to indicate that you have finished giving the suggested input.
Now, provide your reasoning and suggested input below
Reasoning:
"""


def run_eval_output(
    model_name: str,
    dataset_name: str,
    save_name: str,
    override_gen: bool,
    input_file: str,
    output_file: str,
    intermediate_file: str,
    df: pd.DataFrame,
    prompt_column: str,
    prediction_file: str,
    output_column: str = "predicted_output_output",
    max_new_tokens: int = 300,
):
    """
    Common function for running output prediction evaluation.
    """
    df["original_index"] = df.index
    df.to_json(input_file, orient="records", lines=True)
    df = call_infer(
        run_name=save_name,
        dataset_name=dataset_name,
        split="test",
        input_file=input_file,
        output_file=intermediate_file,
        input_column=prompt_column,
        output_column=output_column,
        max_new_tokens=max_new_tokens,
        model=model_name,
        parameters=load_parameters(),
        ignore_checkpoint=override_gen,
    )
    if not os.path.exists(intermediate_file):
        log_warn(
            f"Intermediate file {intermediate_file} was not created. This can happen with OpenAI inference. Run again when batch is done."
        )
        return df
    df = pd.read_json(intermediate_file, orient="records", lines=True)
    parse_errors = 0

    def extract_output(row):
        if "description" in row:
            true_description = row["description"]
        else:
            true_description = row["true_description"]
        response = row[output_column]
        if isinstance(response, list):
            response = response[0]
        if "[STOP]" in response:
            output_part = response.split("[STOP]")[0]
        else:
            output_part = response
        if "Expected Output:" in output_part:
            expected_output = output_part.split("Expected Output:")[1].strip()
            return expected_output
        else:
            return None

    parse_errors = 0
    df["predicted_output"] = None
    for i, row in df.iterrows():
        expected_output = extract_output(row)
        if expected_output is not None:
            df.at[i, "predicted_output"] = expected_output
        else:
            parse_errors += 1
    df = df.groupby(df["original_index"]).agg({"predicted_output": list}).reset_index()
    original_df = pd.read_json(prediction_file, orient="records", lines=True)
    original_df["predicted_output"] = df["predicted_output"]
    original_df.to_json(output_file, orient="records", lines=True)
    log_info(
        f"Saved predicted outputs to {output_file} | Parse errors: {parse_errors}/{len(df)}"
    )
    return original_df



def get_output_model(model_name):
    parameters = load_parameters()
    input_output_model = parameters["input_output_prediction_model_name"]
    if input_output_model == "self":
        input_output_model = model_name
    return input_output_model

def do_predict_output(
    model_name, dataset_name, save_name, override_gen, prediction_column, load_name=None
):
    if load_name is None:
        load_name = save_name

    prediction_file = get_save_paths(dataset_name, load_name)
    save_name = get_output_save_name(save_name)
    input_output_model = get_output_model(model_name)
    output_file = get_save_paths(dataset_name, save_name)
    if os.path.exists(output_file) and not override_gen:
        log_info(
            f"Output file {output_file} already exists, skipping generation. Run with override_gen=True to re-evaluate."
        )
        return
    input_file = output_file.replace(".jsonl", "_input.jsonl")
    intermediate_file = output_file.replace(".jsonl", "_intermediate.jsonl")
    df = pd.read_json(prediction_file, orient="records", lines=True)

    def make_predict_output_prompt(row):
        true_description = None
        if "true_description" in row:
            true_description = row["true_description"]
        else:
            true_description = row["description"]
        predicted_description = row["predicted_description_clean"]
        examples_str = get_all_examples_str(row)
        if prediction_column not in ["prediction", "true"]:
            log_error(
                f"Invalid prediction column: {prediction_column}. Must be either 'prediction' or 'true'."
            )
        use_description = (
            predicted_description
            if prediction_column == "prediction"
            else true_description
        )
        prompt = output_prediction_prompt.replace(
            "[DESCRIPTION]", use_description
        ).replace("[EXAMPLES]", examples_str)
        return prompt

    df["predict_output_prompt"] = df.apply(make_predict_output_prompt, axis=1)
    # now we need to explode the dataframe so that we have one row per example input-output pair, since we want to predict the output for each example input separately
    df = df.explode("test_inputs")
    df["predict_output_prompt"] = df.apply(
        lambda row: row["predict_output_prompt"].replace(
            "[INPUT]", f"{row['test_inputs']}"
        ),
        axis=1,
    )
    run_eval_output(
        model_name=input_output_model,
        dataset_name=dataset_name,
        save_name=save_name,
        override_gen=override_gen,
        input_file=input_file,
        output_file=output_file,
        intermediate_file=intermediate_file,
        df=df,
        prompt_column="predict_output_prompt",
        prediction_file=prediction_file,
        output_column="predicted_output_output",
        max_new_tokens=300,
    )


def run_eval_input(
    model_name: str,
    dataset_name: str,
    save_name: str,
    override_gen: bool,
    input_file: str,
    output_file: str,
    intermediate_file: str,
    df: pd.DataFrame,
    prompt_column: str,
    prediction_file: str,
    target_outputs: pd.Series,
    output_column: str = "predicted_input_output",
    max_new_tokens: int = 300,
):
    """
    Common function for running input prediction evaluation.
    """
    df["original_index"] = df.index
    df.to_json(input_file, orient="records", lines=True)
    print(input_file)
    df = call_infer(
        run_name=save_name,
        dataset_name=dataset_name,
        split="test",
        input_file=input_file,
        output_file=intermediate_file,
        input_column=prompt_column,
        output_column=output_column,
        max_new_tokens=max_new_tokens,
        model=model_name,
        parameters=load_parameters(),
        ignore_checkpoint=override_gen,
    )
    if not os.path.exists(intermediate_file):
        log_warn(
            f"Intermediate file {intermediate_file} was not created. This can happen with OpenAI inference. Run again when batch is done."
        )
        return df
    df = pd.read_json(intermediate_file, orient="records", lines=True)

    def extract_input(row):
        if "description" in row:
            true_description = row["description"]
        else:
            true_description = row["true_description"]
        response = row[output_column]
        if isinstance(response, list):
            response = response[0]
        if "[STOP]" in response:
            input_part = response.split("[STOP]")[0]
        else:
            input_part = response
        if "Suggested Input:" in input_part:
            suggested_input = input_part.split("Suggested Input:")[1].strip()
            return suggested_input
        else:
            return None

    parse_errors = 0
    df["predicted_input"] = None
    for i, row in df.iterrows():
        predicted_input = extract_input(row)
        if predicted_input is not None:
            df.at[i, "predicted_input"] = predicted_input
        else:
            parse_errors += 1
    df = df.groupby(df["original_index"]).agg({"predicted_input": list}).reset_index()
    original_df = pd.read_json(prediction_file, orient="records", lines=True)
    original_df["predicted_input"] = df["predicted_input"]
    original_df["target_outputs"] = target_outputs
    original_df.to_json(output_file, orient="records", lines=True)
    log_info(
        f"Saved predicted inputs to {output_file} | Parse errors: {parse_errors}/{len(df)}"
    )
    return original_df


def get_input_model(model_name):
    parameters = load_parameters()
    input_output_model = parameters["input_output_prediction_model_name"]
    if input_output_model == "self":
        input_output_model = model_name
    return input_output_model

def do_predict_input(
    model_name, dataset_name, save_name, override_gen, prediction_column, load_name=None
):
    if load_name is None:
        load_name = save_name
    prediction_file = get_save_paths(dataset_name, load_name)
    save_name = get_input_save_name(save_name)
    input_output_model = get_input_model(model_name)
    output_file = get_save_paths(dataset_name, save_name)
    if os.path.exists(output_file) and not override_gen:
        log_info(
            f"Output file {output_file} already exists, skipping generation. Run with override_gen=True to re-evaluate."
        )
        return
    input_file = output_file.replace(".jsonl", "_input.jsonl")
    intermediate_file = output_file.replace(".jsonl", "_intermediate.jsonl")
    df = pd.read_json(prediction_file, orient="records", lines=True)
    df["target_outputs"] = None
    for i, row in df.iterrows():
        if "description" in row:
            true_description = row["description"]
        else:
            true_description = row["true_description"]
        func_code = row["test_func_validated"]
        test_inputs = row["test_inputs"]
        target_outputs = []
        try:
            runner = RunTestFunc(func_code)
            for test_input in test_inputs:
                ret, err = runner.run_test_str(test_input)
                target_outputs.append(repr(ret))
        except:
            log_warn(
                f"Could not run test function for row with description: {true_description}. This should never happen."
            )
            continue
        df.at[i, "target_outputs"] = target_outputs

    def make_predict_input_prompt(row):
        if "description" in row:
            true_description = row["description"]
        else:
            true_description = row["true_description"]
        description = row["predicted_description_clean"]
        test_func_str = row["test_func_validated"]
        func_header = get_test_func_header(test_func_str)
        examples_str = get_all_examples_str(row)
        if prediction_column not in ["prediction", "true"]:
            log_error(
                f"Invalid prediction column: {prediction_column}. Must be either 'prediction' or 'true'."
            )
        use_description = (
            description if prediction_column == "prediction" else true_description
        )
        prompt = (
            input_prediction_prompt.replace("[DESCRIPTION]", use_description)
            .replace("[HEADER]", func_header)
            .replace("[EXAMPLES]", examples_str)
        )
        return prompt

    df["predict_input_prompt"] = df.apply(make_predict_input_prompt, axis=1)
    # explode and use target_outputs as the new output to predict the input for
    target_outputs = df["target_outputs"]
    df = df.explode("target_outputs")
    df["predict_input_prompt"] = df.apply(
        lambda row: row["predict_input_prompt"].replace(
            "[OUTPUT]", f"{row['target_outputs']}"
        ),
        axis=1,
    )
    run_eval_input(
        model_name=input_output_model,
        dataset_name=dataset_name,
        save_name=save_name,
        override_gen=override_gen,
        input_file=input_file,
        output_file=output_file,
        intermediate_file=intermediate_file,
        df=df,
        prompt_column="predict_input_prompt",
        prediction_file=prediction_file,
        target_outputs=target_outputs,
        output_column="predicted_input_output",
        max_new_tokens=300,
    )


@click.command()
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def predict_output(model_name, dataset_name, save_name, override_gen):
    do_predict_output(
        model_name,
        dataset_name,
        save_name,
        override_gen,
        prediction_column="prediction",
    )


@click.command()
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def predict_input(model_name, dataset_name, save_name, override_gen):
    do_predict_input(
        model_name,
        dataset_name,
        save_name,
        override_gen,
        prediction_column="prediction",
    )


@click.command()
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def predict_gold_code(model_name, dataset_name, save_name, override_gen):
    do_predict_code(
        model_name,
        dataset_name,
        "gold_" + save_name.split("_", 1)[1],
        override_gen,
        prediction_column="true",
        load_name=save_name,
    )


@click.command()
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def predict_gold_output(model_name, dataset_name, save_name, override_gen):
    do_predict_output(
        model_name,
        dataset_name,
        "gold_" + save_name.split("_", 1)[1],
        override_gen,
        prediction_column="true",
        load_name=save_name,
    )


@click.command()
@click.option("--model_name", type=str, required=True, help="Name of the model to use.")
@click.option("--dataset_name", type=str, required=True, help="Name of the dataset.")
@click.option(
    "--save_name", type=str, default=None, help="Name to save the predictions under."
)
@click.option(
    "--override_gen", is_flag=True, help="Whether to override existing generation."
)
def predict_gold_input(model_name, dataset_name, save_name, override_gen):
    do_predict_input(
        model_name,
        dataset_name,
        "gold_" + save_name.split("_", 1)[1],
        override_gen,
        prediction_column="true",
        load_name=save_name,
    )


@click.group()
def cli():
    pass


cli.add_command(run_finetuned, name="finetuned")
cli.add_command(run_zeroshot, name="zeroshot")
cli.add_command(predict_code, name="code")
cli.add_command(predict_output, name="output")
cli.add_command(predict_input, name="input")
cli.add_command(predict_gold_code, name="gold_code")
cli.add_command(predict_gold_output, name="gold_output")
cli.add_command(predict_gold_input, name="gold_input")


if __name__ == "__main__":
    cli()
